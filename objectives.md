# BEVDrive — Development Objectives & AI Agent Execution Plan

> **Document type:** Implementation specification / acceptance contract  
> **Status:** Planned — only the existing YOLO 2D VRU detector is considered available until the agent verifies its files/checkpoint.  
> **Scope:** Research prototype for camera–LiDAR autonomous driving perception, prediction and simulation-based planning. **Not approved for real-vehicle control.**  
> **Last updated:** 2026-10-19

## 0. Project mission and architecture decision

**Mission:** Build an extensible camera + LiDAR autonomous-driving research stack, starting with unified 3D object detection (vehicles + vulnerable road users/VRUs), then 3D tracking, behavior/motion prediction, and semantic occupancy + risk-aware planning.

**Fixed architecture decisions:**

1. **Main route:** camera encoder + LiDAR encoder → shared BEV features (BEVFusion) → unified multi-class 3D detection → 3D MOT → motion/behavior prediction. Occupancy/map features run as *parallel consumers of the shared spatial representation*, not as a stage mechanically produced by tracking boxes.
2. **Vehicle and VRU:** initially share the **same detector/backbone and 3D head**; separate classes, error analysis and downstream policy. Add a VRU-specific head **only if ablation demonstrates measurable benefit** with acceptable cost.
3. **Existing YOLO26m 2D checkpoint trained on BDD100K:** keep as a **separate camera-only baseline** in Phase 1. It is **not required** for BEVFusion inference. An optional YOLO-assisted fusion branch is a later **controlled experiment**, *not a default production dependency*.
4. **Dataset boundary:** BDD100K-YOLO images/labels do not provide synchronized LiDAR. Use nuScenes camera + LiDAR + sensor calibration + ego poses for unified 3D perception/tracking. PIE/JAAD can support video-based pedestrian intention work, but **cannot be silently treated as nuScenes-style 3D intention ground truth**.
5. **BEV ≠ occupancy:** BEV is a coordinate/feature representation; a semantic 3D occupancy head requires its own supervision or defensible self-supervision. A 2D BEV projection is not automatically full 3D occupancy.
6. **End-to-end (E2E) terminology:** Phase 4 first delivers a **modular, closed-loop simulated stack**. An optionally jointly trained sensor-to-planning model is a *stretch research objective*, not something achieved by merely connecting modules.
7. **Causal/online behavior:** inference at time `t` may use only data available at or before `t`; future GT is permitted **only for training labels/evaluation**, never as inference features.

### Target architecture

```text
                         [Multi-camera]             [LiDAR]
                               |                       |
                        Image Encoder            Point Encoder
                               |                       |
                       Image-to-BEV             LiDAR-to-BEV
                               \                       /
                                \---- BEVFusion -------/
                                          |
                              Shared Spatial Features
                                /                 \
                     Unified 3D Detector      Map/Occupancy Head
                     (Vehicle + VRU)          (parallel module)
                                |                 |
                        3D Multi-Object MOT        |
                                |                 |
                        Agent State Histories     |
                                \                 /
                             Behavior + Motion Prediction
                                          |
                              Collision/Risk Evaluation
                                          |
                             Simulation-based Planning
                                          |
                             Safety Monitor / Evaluation

  Existing BDD100K YOLO26m --> independent 2D baseline / error analysis
  Optional experiment (NOT default): YOLO proposals --> cross-view association
```

## 1. AI coding agent operating contract — READ FIRST

The agent **MUST** follow these instructions for every change.

### 1.1 Before coding

- Inspect the repository and report which files/modules already exist; **do not overwrite the existing YOLO training pipeline or checkpoints**.
- Check hardware (`nvidia-smi`, GPU model/VRAM, CPU RAM, disk), OS, Python, CUDA, available datasets/checkpoints. Record to `docs/environment.md`.
- Prefer a Linux + NVIDIA CUDA environment. **Do not invent a working version combination** for PyTorch / MMCV / MMDetection3D; pick an officially compatible, tested matrix from the selected implementation and lock it in `environment.yml`/`requirements-lock.txt` and `docs/environment.md`.
- Prefer the **MMDetection3D BEVFusion project** as the first integration candidate. The **MIT-Han-Lab BEVFusion original repository is archived and depends on an older stack**; isolate it in Docker/conda only when explicitly choosing it. Verify current config, checkpoint link, license and command names *before* documenting commands as working.
- Confirm dataset license/access, directory layout, hashes/manifest, annotation split and correct calibration. Never fetch multi-GB datasets, install system-level CUDA or use paid cloud resources without explicit user authorization.
- If input data or GPU is missing, implement **synthetic/CPU unit tests and sample-adapter smoke tests first**; report clearly what could not be tested. Never present placeholder output as a trained result.

### 1.2 Implementation standards

- Python with `src/` package layout. Typed contracts (`dataclass`/`pydantic` or `TypedDict`), type hints, logging, errors with actionable messages, config-driven operation (YAML/MMEngine), deterministic seeds and minimal reproducible examples.
- No hardcoded absolute dataset paths, API keys, local usernames, GPU IDs, credentials, or weights in source files. Paths belong in `.env.example` and `configs/*.yaml`.
- Implement independently testable components; avoid copying a full third-party research repo into version control without license and provenance records.
- Preserve **timestamps in seconds**, **translation in meters**, **velocity in m/s**, **angles in radians**; always provide `coordinate_frame`, camera/LiDAR transforms and units.
- Scene/sequence IDs scope all track IDs. Reset state at scene boundaries. Maintain visible/missing-measurement masks and uncertainty.
- Keep raw data, checkpoints and heavy generated artifacts out of Git via `.gitignore`; use configurable storage locations and a manifest with version, hash and source URL.
- Build `--dry-run`, `--device cpu|cuda`, `--limit-samples`, `--seed`, `--output-dir` for long-running commands where applicable.
- Long training jobs must **resume** from full checkpoints: model, optimizer, LR scheduler, AMP scaler (if used), epoch/step, best metric, config and random states. Use atomic saves and validate checkpoint loading. Data sampler progress should be restored where supported; otherwise document that an interrupted epoch will restart.
- Each experiment logs run ID, exact dataset split, model/config/checkpoint hashes, commit SHA, device and metrics. Use TensorBoard or W&B (W&B only if configured; otherwise local JSON/CSV).
- Use tests (`pytest`), formatting/lint (`ruff`), and CI CPU smoke tests. Do not require GPU or private data for core CI.
- Never report guessed accuracy/latency. If no benchmark is run, mark `NOT_MEASURED`.

### 1.3 Execution protocol and phase gates

For each phase:

1. Create/update `docs/phaseN_plan.md` with verified dependencies, available inputs and runnable commands.
2. Implement smallest **vertical slice** (one sample → valid output) before large training.
3. Add unit/integration tests + visual sanity checks, then deterministic mini smoke run.
4. Run full training/evaluation **only with actual accessible data/resources** and only on the correct split.
5. Save results to `reports/phaseN/` (`metrics.json`, `summary.md`, `figures/`, `run_manifest.json`).
6. Update `docs/progress.md` with completed checklist, blockers, commands and next action.
7. Do not mark the phase `DONE` until *all required acceptance criteria* pass. Never silently skip prerequisites or substitute synthetic data for real evaluation.

**Status convention:** `TODO` → `IN_PROGRESS` → `BLOCKED` or `DONE`. Track incomplete items without inventing success.

## 2. Planned technology stack

| Layer | Default choice | Purpose / rule |
|---|---|---|
| Language / ML | Python, PyTorch, torchvision | DL training and inference; versions pinned after hardware audit |
| 3D perception | MMDetection3D, MMEngine, BEVFusion | Unified camera–LiDAR BEV detection; use verified config/checkpoint |
| LiDAR-only baseline | CenterPoint or supported LiDAR-only BEVFusion/TransFusion-L config | Isolate benefit of fusion |
| Existing 2D baseline | Ultralytics YOLO26m checkpoint (existing) | Evaluate BDD100K detector separately on same target frames |
| Datasets / eval | nuScenes + `nuscenes-devkit` | Calibrated multi-sensor samples, official 3D detection and tracking evaluation |
| Point cloud / geometry | NumPy, SciPy, Open3D (visualization), PyQuaternion, OpenCV | Transformations, projections, visualization; avoid Open3D in model hot path |
| 3D tracking | Kalman filter + Hungarian, configurable class-aware association | Explainable initial online MOT baseline |
| Behavior / forecast | PyTorch GRU/LSTM baseline, optional Transformer | Sequential crossing/action and multimodal motion forecast |
| Behavior dataset | PIE/JAAD (separate domain), nuScenes tracks/maps for trajectory | Only use labels actually provided; define task-specific splits |
| Occupancy | Occ3D-nuScenes + verified occupancy baseline | 3D voxel semantics, free/occupied/unknown where labels support it |
| Planning & simulation | NumPy/SciPy/Shapely for baseline; optional CARLA or nuPlan later | Collision checking and closed-loop evaluation |
| Config / CLI | YAML + MMEngine for 3D training; `argparse`/Typer for repo CLI | Reproducibility, no implicit magic paths |
| Experiment management | TensorBoard; optional W&B | Metrics, configs, checkpoints, compare runs |
| Quality | pytest, ruff, mypy optional, GitHub Actions | CPU CI, style, schema tests |
| Packaging | `pyproject.toml`, lockfile, Dockerfile | Reproducible local development |

**Dependency policy:** This is a *component selection*, **not** a claim that all packages install together on the latest Python/PyTorch. Separate compatible environments or containers for legacy 3D repos if needed. `docs/environment.md` is the verified source of truth.

## 3. Data contracts shared by all phases

All modules MUST use versioned schemas under `src/bevdrive/contracts/`.

### 3.1 `SensorFrame` (per timestamp)

```python
SensorFrame = {
    "scene_id": "scene-...",
    "sample_token": "...",
    "timestamp_us": 0,                       # integer from dataset
    "camera_images": {"CAM_FRONT": "..."},  # sensor-id -> path or lazy image
    "lidar_path": "...",
    "intrinsics": {"CAM_FRONT": "3x3"},
    "T_cam_from_lidar": {"CAM_FRONT": "4x4"},
    "T_world_from_ego": "4x4",             # explicit timestamp
    "T_ego_from_sensor": {"LIDAR_TOP": "4x4"},
    "valid_sensors": {"CAM_FRONT": True, "LIDAR_TOP": True}
}
```

*Contract examples illustrate field semantics, not necessarily exact serialized NumPy types.* Synchronize and transform sensor poses at acquisition timestamps; do not assume all sensor captures are perfectly simultaneous.

### 3.2 `Detection3D`

```python
Detection3D = {
    "scene_id": "...", "sample_token": "...", "timestamp_us": 0,
    "class_name": "pedestrian", "score": 0.0,
    "center_xyz_m": [0.0, 0.0, 0.0],
    "size_lwh_m": [0.0, 0.0, 0.0],
    "yaw_rad": 0.0,
    "velocity_xy_mps": None,               # optional detector prediction
    "coordinate_frame": "ego_at_timestamp",
    "covariance": None,                    # explicit null if unknown
    "source": "bevfusion"
}
```

### 3.3 `TrackedAgent3D`

Extends `Detection3D`: `track_id`, `track_age`, `measurement_status`, `state_world_xyz`, `velocity_world_xy_mps`, `state_covariance`, `history[]`, `last_observed_timestamp_us`. **Do not assert velocity comes directly from a single LiDAR sweep.**

### 3.4 `PredictionResult`

```python
PredictionResult = {
    "scene_id": "...", "track_id": "...", "timestamp_us": 0,
    "forecast_frame": "ego_at_prediction_time",
    "future_timestamps_sec": [0.5, 1.0, 1.5],
    "trajectories_xy_m": [[[0, 0], [0, 0], [0, 0]]],
    "mode_probabilities": [1.0],
    "crossing_probability_within_2s": None,
    "prediction_valid": True
}
```

`crossing_probability_within_2s = None` when model is untrained, required annotations unavailable, or task is inapplicable. It must not be fabricated from bounding boxes or from the pedestrian class.

### 3.5 `OccupancyResult` / `RiskResult`

Occupancy: `voxel_origin_xyz_m`, `voxel_resolution_xyz_m`, `grid_shape`, `semantic_classes`, `occupancy_probabilities`/labels, visibility/unknown mask, reference frame, horizon. Risk: forecast time, ego candidate trajectory, occupancy conflict or agent overlap, uncertainty and explicit non-safety-certified warning flag.

## 4. PHASE 1 — Unified 3D Detection (Camera + LiDAR)

**Status:** `TODO`  
**Purpose:** Create one calibrated, shared BEV-based 3D detector for **vehicle + VRU**; establish trustworthy baselines and output consistent 3D boxes for later tracking.  
**NO YOLO in main inference path.** Existing YOLO is baseline-only in this phase.

### Inputs

- nuScenes `v1.0-mini` for integration/debug; `v1.0-trainval` for meaningful training/validation when available.
- RGB multi-camera images, LiDAR point clouds, timestamps, sensor calibration, ego poses and 3D box labels.
- Verified BEVFusion configuration and compatible pretrained weights; optional LiDAR-only detector checkpoint.
- Existing BDD100K YOLO26m weights for a separate 2D comparison when paths are provided.

### Technical tasks (execute in order)

- **P1.1 — Dataset adapter:** implement nuScenes loader, class mapping, scene-safe split handling and metadata validation. Validate each camera↔LiDAR transform by projecting sparse points/3D GT boxes onto images; save overlays and BEV screenshots. Detect missing sensors and timestamp mismatches; fail loudly or mark invalid.
- **P1.2 — Reproducible runtime:** create install/setup instructions with pinned tested dependency matrix. Provide `device` and `limit-samples` options, sample config and one-command smoke test.
- **P1.3 — LiDAR-only baseline:** infer with an available CenterPoint/TransFusion-L implementation; export `Detection3D` and visualizations.
- **P1.4 — BEVFusion main model:** run pretrained camera+LiDAR 3D detection; verify image backbone, image-to-BEV, LiDAR features, BEV fusion and multi-class head are configured as expected. Use same nuScenes split for comparison.
- **P1.5 — Independent 2D baseline:** run existing YOLO on matching nuScenes camera frames; adapt label taxonomy and report **2D camera-space metrics only** where GT supports it. Do **not** compare YOLO 2D mAP numerically with nuScenes 3D mAP. Project 3D GT/predictions to image for qualitative matched error analysis, accounting for truncation and camera visibility.
- **P1.6 — VRU-focused analysis:** report per-class performance for pedestrian, bicycle, motorcycle and appropriate vehicle classes; stratify by range/occlusion/visibility **only where labels or a documented proxy support this**. Generate false-positive/false-negative galleries and 3D center/size/yaw errors.
- **P1.7 — Optional fine-tuning:** enable only after baseline evaluation. Train/resume on training split; optimize with documented configs; evaluate on held-out validation; never tune on test labels.
- **P1.8 — Optional YOLO auxiliary experiment (NOT phase gate):** implement proposal/confirmation branch and geometry-aware 2D↔3D association with duplicate handling; compare without/with YOLO at same evaluation budget. Only promote if VRU recall/precision and latency tradeoff support it; no mandatory reliance on YOLO for valid 3D detections.

### Required repository files

```text
src/bevdrive/datasets/nuscenes_adapter.py
src/bevdrive/geometry/transforms.py
src/bevdrive/geometry/projection.py
src/bevdrive/contracts/detection.py
src/bevdrive/perception/bev_detector.py
src/bevdrive/perception/lidar_baseline.py
src/bevdrive/baselines/yolo2d.py
src/bevdrive/evaluation/detection.py
src/bevdrive/visualization/overlays.py
configs/datasets/nuscenes_mini.yaml
configs/detection/bevfusion.yaml
configs/detection/lidar_only.yaml
configs/baselines/yolo2d.yaml
scripts/phase1_smoke.py
scripts/phase1_evaluate.py
scripts/phase1_train.py
reports/phase1/
tests/test_calibration_projection.py
tests/test_detection_schema.py
```

### CLI contract (agent must implement, not assumed pre-existing)

```bash
python scripts/phase1_smoke.py --config configs/detection/bevfusion.yaml --limit-samples 2 --output-dir outputs/phase1_smoke
python scripts/phase1_evaluate.py --config configs/detection/bevfusion.yaml --split val --output-dir reports/phase1
python scripts/phase1_train.py --config configs/detection/bevfusion.yaml --resume auto --output-dir runs/phase1
pytest -q tests/test_calibration_projection.py tests/test_detection_schema.py
```

### Deliverables

1. Calibrated nuScenes loader + alignment overlays and BEV visualization.
2. LiDAR-only and camera+LiDAR model adapters with working checkpoint inference.
3. Unified multi-class 3D detections (`Detection3D`) with correct coordinates.
4. YOLO 2D baseline report, independently labeled and evaluated.
5. `reports/phase1/metrics.json`, per-class tables, run manifest, failure gallery and `summary.md`.
6. Resumable training + evaluation script if training is undertaken.

### Definition of Done / acceptance gate

- [ ] Automated projection/coordinate-transform tests pass (including transform inverse and round-trip tolerance).
- [ ] Two or more **real and synchronized** nuScenes samples produce nonempty or correctly empty 3D detections without schema/NaN errors; rendered camera+BEV overlays have been manually inspected.
- [ ] LiDAR-only and BEVFusion inference are both executed or explicitly marked `BLOCKED` with reason. Phase **cannot** be `DONE` without real camera+LiDAR inference.
- [ ] Official nuScenes 3D detection evaluator runs on a legitimate held-out validation subset/full set as available; dataset size and split are disclosed. Mini smoke metrics are labeled **smoke only**, never called leaderboard numbers.
- [ ] End-to-end per-sample inference latency is measured on named hardware or marked `NOT_MEASURED` if no compatible device.
- [ ] YOLO remains optional: BEVFusion can run with YOLO disabled.

**Phase 1 exit artifact:** `outputs/phase1/detections.jsonl` (or canonical serialized equivalent) + BEVFusion checkpoint metadata + per-class evaluation report.

## 5. PHASE 2 — 3D Multi-Object Tracking (MOT)

**Status:** `TODO`  
**Dependency:** Phase 1 real `Detection3D` interface.  
**Purpose:** Assign stable identity to vehicles and VRUs, estimate their motion in physical units, and create history for future prediction.

### Inputs

- Phase 1 detections in chronological order grouped by nuScenes scene.
- Ego poses/transforms and timestamps; optional detector velocities and confidence.
- nuScenes tracking ground truth / official tracking evaluation interface.

### Technical tasks

- **P2.1 — Reference frames:** transform 3D detections into a consistent world/odometry frame at matching timestamps; run filter there; convert outputs to the requested ego frame. Interpolate ego pose only where mathematically valid and within supported timestamp bounds.
- **P2.2 — Motion model:** implement constant-velocity Kalman baseline (`[x,y,vx,vy]` minimum, with z, size, yaw handled explicitly), variable `dt`, innovation gating and covariance propagation. Add optional constant-turn or EKF only after baseline.
- **P2.3 — Data association:** Hungarian assignment with class compatibility and 3D center distance / BEV IoU / Mahalanobis distance, configurable gates and confidence handling; avoid arbitrary match of detections far apart.
- **P2.4 — Track lifecycle:** tentative/confirmed/lost/deleted states; `min_hits`, `max_age_seconds`, occlusion survival, confidence decay and `observed_mask`. Reset on scene change. Do not mix IDs across scenes.
- **P2.5 — Streaming output:** chronological `TrackedAgent3D` histories; coordinate frame and covariance always specified. Provide comparison run with ground-truth 3D detections as **oracle diagnostic only** (clearly separate from actual detector tracking).
- **P2.6 — Benchmark:** nuScenes tracking metrics using official devkit (AMOTA/AMOTP and ID-switch metrics); additionally report VRU-specific fragmentation, velocity error when measurable and ID consistency.
- **P2.7 — Failure cases:** crossing paths, occlusion, stationary pedestrians while ego moves, poor LiDAR returns, time gaps and objects entering/leaving field of view.

### Tech stack

PyTorch detector outputs, NumPy, SciPy (`linear_sum_assignment`), Kalman filter implemented/tested in-house or well-documented package; nuScenes tracking evaluator; pytest.

### Required files / CLI

```text
src/bevdrive/tracking/kalman.py
src/bevdrive/tracking/association.py
src/bevdrive/tracking/track_manager.py
src/bevdrive/tracking/coordinate_utils.py
src/bevdrive/evaluation/tracking.py
configs/tracking/kalman_hungarian.yaml
scripts/phase2_track.py
scripts/phase2_evaluate.py
tests/test_tracker_scene_reset.py
tests/test_tracker_ego_motion.py
tests/test_tracker_occlusion.py
reports/phase2/
```

```bash
python scripts/phase2_track.py --config configs/tracking/kalman_hungarian.yaml --detections outputs/phase1/detections.jsonl --output-dir outputs/phase2
python scripts/phase2_evaluate.py --predictions outputs/phase2/tracks.json --split val --output-dir reports/phase2
pytest -q tests/test_tracker_scene_reset.py tests/test_tracker_ego_motion.py tests/test_tracker_occlusion.py
```

### Deliverables

- Online 3D tracking engine, config, chronological track output and visualized motion history on BEV.
- Official tracking metrics and class-specific VRU error analysis.
- Test evidence for ego-motion compensation and ID tracking across occlusion/scene reset.
- `reports/phase2/{metrics.json,summary.md,run_manifest.json}`.

### Definition of Done

- [ ] Synthetic test: stationary world pedestrian remains near zero world-velocity while ego vehicle moves.
- [ ] Synthetic test: tracks remain uniquely identified through short occlusion; reset correctly on new scene.
- [ ] Variable-`dt` and non-monotonic timestamp tests pass (reject invalid timestamps).
- [ ] Valid online nuScenes tracking output submitted to local official evaluator, with real detections; AMOTA/related metrics reported without fabrication.
- [ ] 3D track history available to Phase 3 with `track_id`, velocity, covariance and observation validity.

**Phase 2 exit artifact:** `outputs/phase2/tracks.json` / `tracks.jsonl` with versioned schema and measured metrics.

## 6. PHASE 3 — VRU Behavior & Multi-Modal Motion Prediction

**Status:** `TODO`  
**Dependency:** Phase 2 metric tracking + scene context.  
**Purpose:** Predict **what each VRU is doing / is likely to do**, and **where the VRU may go**, with calibrated uncertainty. Do not confuse crossing *state* with crossing *intention*.

### Task definitions (must be explicit)

- **Action/state recognition:** classify current observable state only when labels exist (e.g., standing / walking / running / crossing-now). Do not infer unobservable gaze or pose from 3D center alone.
- **Crossing intention:** `P(onset of road crossing within H seconds | history up to t, scene)`, restricted to eligible non-crossing pedestrian observations. Use a time horizon defined in seconds and a clear event-onset annotation protocol.
- **Trajectory forecasting:** predict `K` plausible future agent trajectories in metric ego/world coordinates with mode probabilities and optional uncertainty. Start with a constant-velocity baseline; learn GRU/LSTM then optional Transformer.
- **Safety-relevant context:** crosswalk/sidewalk/road boundaries, ego velocity/pose, nearest-agent interactions, and optional synchronized camera ROI/pose features.

### Dataset separation (critical)

1. **nuScenes**: train/test **3D trajectory forecasting** from tracks/annotations with the official or clearly documented protocol; use maps and ego poses. nuScenes 3D tracks are **not automatically labeled with pedestrian future-crossing intentions or walking/running actions**.
2. **PIE/JAAD**: train and evaluate crossing intention / camera-action models with their actual provided annotations. Data are video/image-space; no automatic 3D coordinates or alignment with nuScenes LiDAR exists.
3. When integrating PIE-trained camera behavior into nuScenes BEV inference, treat it as **cross-domain transfer requiring validation**. If calibrated crossing output on nuScenes is unsupported, produce `null`/`UNKNOWN`, not made-up probabilities.
4. Define splits at **video/scene level**, never random windows crossing train/validation/test boundaries. Prediction labels may use future annotations; inference features must never use those future observations.

### Technical tasks

- **P3.1 — Feature builder:** for each `track_id`, construct uniformly timestamped historical state sequence (`x,y,vx,vy,yaw,object_dims,covariance,observed_mask`), map features and ego features. Resample/interpolate with validity flags; never pretend missing measurements are observed.
- **P3.2 — Baselines:** constant velocity/acceleration trajectory, logistic regression or MLP crossing baseline when labeled data permit. Document horizons, observation duration and label eligibility.
- **P3.3 — Behavior model:** GRU temporal encoder with optional action head and binary intention head; masked losses for missing labels; class-balanced evaluation, threshold tuning on validation only, PR-AUC/F1/Recall plus Brier/ECE where appropriate.
- **P3.4 — Future motion model:** GRU encoder-decoder `K=1` baseline then `K>1` multimodal predictor; output future timestamps, candidate trajectories, normalized probabilities, confidence/uncertainty.
- **P3.5 — Context/interaction ablations:** motion-only vs +map vs +ego vs +visual features, keeping splits and evaluation protocol identical. VRU-specific failure analysis near crosswalks, occlusion and sudden starts/stops.
- **P3.6 — Integration:** publish `PredictionResult` for each tracked VRU; define clear fallbacks for short/invalid history, unlabeled tasks and missing camera ROI.
- **P3.7 — Validation:** trajectory minADE/minFDE, miss rate (with reported threshold/protocol), crossing PR-AUC/F1, prediction lead time and calibration; disaggregate by class and situation.

### Tech stack

PyTorch, NumPy, pandas/Parquet for sequence data, nuScenes devkit + map API, PIE/JAAD readers, scikit-learn metrics, optional torchvision pose/video encoder. Use GT trajectories for teacher-forced diagnostic **separately** from true tracked-input evaluation.

### Required files / CLI

```text
src/bevdrive/prediction/features.py
src/bevdrive/prediction/datasets/nuscenes_tracks.py
src/bevdrive/prediction/datasets/pie_adapter.py
src/bevdrive/prediction/cv_baseline.py
src/bevdrive/prediction/gru_trajectory.py
src/bevdrive/behavior/gru_intention.py
src/bevdrive/behavior/labels.py
src/bevdrive/evaluation/prediction.py
configs/prediction/trajectory_gru.yaml
configs/behavior/crossing_gru.yaml
scripts/phase3_build_sequences.py
scripts/phase3_train.py
scripts/phase3_evaluate.py
tests/test_prediction_no_future_leak.py
tests/test_prediction_scene_split.py
reports/phase3/
```

```bash
python scripts/phase3_build_sequences.py --dataset nuscenes --split train --output-dir outputs/phase3/sequences
python scripts/phase3_train.py --task trajectory --config configs/prediction/trajectory_gru.yaml --resume auto --output-dir runs/phase3_trajectory
python scripts/phase3_evaluate.py --task trajectory --split val --output-dir reports/phase3
pytest -q tests/test_prediction_no_future_leak.py tests/test_prediction_scene_split.py
```

**Agent must add a separate `--dataset pie --task crossing` training/evaluation path before claiming the crossing model is complete.** Ensure camera/model features and labels are obtainable at runtime.

### Deliverables

- Sequence generation pipeline with class/scene/time validity masks.
- Reproducible trajectory baseline and trained motion model; future trajectories in metric coordinates.
- Distinct crossing-intention model/report on an appropriately labeled dataset; uncertainty and domain-transfer limitations documented.
- Context ablation table, trajectory visualizations, near-crossing demos.
- `reports/phase3/{metrics.json,summary.md,run_manifest.json}` and resumable checkpoints.

### Definition of Done

- [ ] Unit test proves any change to observations **after** time `t` cannot affect inference input or output at `t`.
- [ ] Sequence train/val/test grouping is scene/video-safe; labels and horizon documented in seconds.
- [ ] Constant-velocity baseline and learned trajectory predictor evaluated on held-out metric trajectories with protocol-consistent ADE/FDE/miss rate.
- [ ] Crossing metrics are measured **only** using valid intention labels; no claim that nuScenes supplies native crossing-intention ground truth.
- [ ] No missing-label, missing-depth or out-of-domain input yields fabricated high-confidence behavior prediction.
- [ ] Tracks → predictions integration runs on at least one real recorded scene, reporting invalid/unknown predictions separately.

**Phase 3 exit artifact:** time-indexed per-VRU predicted trajectories and behavior estimates with explicit validity; evaluation and limitations report.

## 7. PHASE 4 — Occupancy, Collision Risk & Planning-Oriented E2E Research

**Status:** `TODO`  
**Dependency:** validated Phase 1 spatial features, Phase 2 states, Phase 3 forecasts; simulator/mapping data must be available.  
**Purpose:** Understand free/occupied/unknown 3D space, anticipate future hazards and plan candidate ego trajectories inside a **simulation-only** loop. Support eventual joint learning without claiming on-road autonomy.

### Scope and architecture

- **Parallel occupancy/map branch:** read shared BEV/camera-LiDAR spatial features (or a verified separate occupancy baseline) to predict semantic 3D voxels and/or BEV drivable-area map.
- **Dynamic prediction:** future agent occupancy can initially be rasterized from tracked predicted trajectories *with uncertainty*. Do not claim full learned future occupancy/occupancy flow without implementing and evaluating that model.
- **Risk engine:** evaluate time-aligned ego + agent occupancy intersection, closest approach, TTC only where definition applies, stopping feasibility, input confidence and worst-case/uncertainty.
- **Planner:** start with rule-based/lattice candidate trajectory generator (lane-follow, slow, yield, stop), kinematic feasibility, collision checking and cost-based selection. Connect to simulation later.
- **E2E stretch goal:** jointly optimize compatible perception/prediction/planning components using a verified differentiable training route and an appropriate dataset/simulator. Do **not** label a modular stack as end-to-end trained.

### Datasets and validation

- Use **Occ3D-nuScenes** or another verified occupancy annotation dataset for quantitative occupancy training/evaluation; nuScenes 3D boxes alone do not provide voxel-level semantic GT.
- Use nuScenes map extension for drivable area, lane, pedestrian crossing and road topology where covered. Understand that a map is a prior and is not a substitute for live occupancy sensing.
- Use a supported closed-loop simulator, e.g. CARLA or a verified planning benchmark; nuScenes recorded scenes alone are **open-loop** and do not become reactive closed-loop simulation automatically.
- No physical steering or braking commands to an actual vehicle. Manual review of hazardous examples; explicit `sim_only: true` default and runtime guard.

### Technical tasks

- **P4.1 — Occupancy contract and loader:** voxel grid orientation, origin, resolution, semantic label mapping, unknown/visibility masks; verify calibration and GT transform using known geometric samples.
- **P4.2 — Occupancy baseline:** load pretrained occupancy model if compatible; otherwise train minimal occupancy head with supervision. Compare with map/voxel baseline. Track occupied-vs-free precision/recall and semantic mIoU with visibility policy reported.
- **P4.3 — Dynamic rasterization:** from Phase 3 `K` forecast modes, create time-indexed occupancy with mode weights; model spatial footprint + uncertainty and flag unsupported voxels/unknown regions.
- **P4.4 — Collision risk:** implement continuous/discrete time-indexed footprint overlap; motion-consistent distances; near-miss and TTC calculations with clear conditions. Define fail-closed simulation fallback for missing perception/prediction.
- **P4.5 — Ego planner:** candidate trajectory generator, kinematic constraints, goal progress, road boundary compliance, collision cost, comfortable braking and yield/stop decisions. **Tuning is research-only**, not safety certified.
- **P4.6 — Simulation loop:** connect `Sensor/Recorded data → Perception → Tracking → Prediction → Occupancy/Risk → Planner → Simulated ego`; log all intermediate outputs and replayable scenario seeds. Start with deterministic scripted scenarios, then supported reactive closed-loop simulator.
- **P4.7 — Evaluations:** occupancy semantic mIoU/free-space errors, collision/near-miss rate, intervention/failure count, route progress, off-road rate, false alerts, detection-to-decision latency and uncertainty stress tests. Include occluded pedestrian crossing, late braking, cyclist turn, sensor dropout and calibration perturbation.
- **P4.8 — E2E (optional):** only after modular baselines are reproducible, propose a separately scoped joint-training experiment, with clear loss definition, sensor-to-trajectory gradient path, data, compute budget and closed-loop evaluation. This is **not a required Phase 4 gate**.

### Tech stack

PyTorch + verified BEV/occupancy model, Occ3D-nuScenes, NumPy/SciPy/Shapely, nuScenes map SDK, simulation adapters (CARLA or equivalent only when installed and approved), Matplotlib/Open3D for visualization, pytest.

### Required files / CLI

```text
src/bevdrive/occupancy/dataset.py
src/bevdrive/occupancy/model.py
src/bevdrive/occupancy/rasterizer.py
src/bevdrive/risk/collision_checker.py
src/bevdrive/risk/risk_engine.py
src/bevdrive/planning/trajectory_generator.py
src/bevdrive/planning/costs.py
src/bevdrive/planning/planner.py
src/bevdrive/simulation/base_adapter.py
src/bevdrive/simulation/scenario_runner.py
configs/occupancy/occ3d.yaml
configs/planning/sim_only.yaml
scripts/phase4_occupancy_eval.py
scripts/phase4_run_scenarios.py
tests/test_occupancy_axes.py
tests/test_collision_overlap.py
tests/test_planner_safety_guard.py
reports/phase4/
```

```bash
python scripts/phase4_occupancy_eval.py --config configs/occupancy/occ3d.yaml --split val --output-dir reports/phase4
python scripts/phase4_run_scenarios.py --config configs/planning/sim_only.yaml --output-dir outputs/phase4_scenarios
pytest -q tests/test_occupancy_axes.py tests/test_collision_overlap.py tests/test_planner_safety_guard.py
```

### Deliverables

- Semantic occupancy/BEV map module with real GT validation and visualizations.
- Dynamic occupancy overlay generated from predicted trajectories and uncertainty.
- Deterministic risk engine with testable spatial–temporal collision checks.
- Sim-only planner and a replayable scenario runner (plus CARLA/other simulator adapter if available).
- Evaluated scenario matrix and regression tests with performance metrics and known limitations.
- `reports/phase4/{metrics.json,summary.md,run_manifest.json,safety_limitations.md}`.

### Definition of Done

- [ ] Occupancy prediction and evaluation are run on matching, correctly transformed GT labels; unknown/invalid regions are handled explicitly.
- [ ] Toy scene tests: non-intersecting paths cause no collision; intersecting geometry at different times is **not** a collision; simultaneous occupied overlap is identified.
- [ ] Planner respects modeled kinematic and drivable-area constraints in synthetic tests; logs why a trajectory was selected.
- [ ] Sim-only guard blocks any physical actuator command; no unreviewed real-vehicle hardware integration.
- [ ] At least three deterministic scenarios (crossing pedestrian, cyclist turning, occluded re-appearance) can be replayed and produce complete logs; quantitative closed-loop claims require a real reactive simulator.
- [ ] Full pipeline reports measured latency, failure modes, uncertainty behavior and non-certified status.

**Phase 4 exit artifact:** modular simulation-ready perception-to-planning prototype, not a road-legal/self-driving product.

## 8. Repository structure to maintain

```text
BEVDrive/
├── objectives.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── environment.yml                  # version-pinned after audit
├── configs/
│   ├── datasets/
│   ├── detection/
│   ├── baselines/
│   ├── tracking/
│   ├── behavior/
│   ├── prediction/
│   ├── occupancy/
│   └── planning/
├── src/bevdrive/
│   ├── contracts/
│   ├── datasets/
│   ├── geometry/
│   ├── perception/
│   ├── baselines/
│   ├── tracking/
│   ├── behavior/
│   ├── prediction/
│   ├── occupancy/
│   ├── risk/
│   ├── planning/
│   ├── simulation/
│   ├── evaluation/
│   └── visualization/
├── scripts/
├── tests/
├── docs/
│   ├── architecture.md
│   ├── environment.md
│   ├── progress.md
│   ├── data_contracts.md
│   ├── dataset_licenses.md
│   └── phaseN_plan.md               # one actual file for each phase
├── reports/phase1/ ... phase4/
├── outputs/                          # gitignored
├── runs/                             # gitignored
├── weights/                          # gitignored
└── data/                             # gitignored; external symlinks OK
```

Use the **smallest maintainable amount of code**. Do not pre-create dozens of empty modules and claim functionality; implement as each phase needs them.

## 9. Benchmarking and promotion rules

**Comparable experiments must use the same split/scene list, hardware, resolution limits, class taxonomy and evaluation protocol.** Distinguish 2D mAP from 3D mAP/NDS. Track class-level safety metrics, not only aggregate scores.

| Experiment | Purpose | When |
|---|---|---|
| E1 — YOLO camera-only (2D) | Prior work/visual error reference | P1 baseline |
| E2 — LiDAR-only 3D detector | Geometry baseline | P1 required |
| E3 — BEVFusion camera+LiDAR 3D | Main perception architecture | P1 required |
| E4 — BEVFusion + VRU-specific head | Optional recall-focused optimization | After P1 metrics |
| E5 — BEVFusion + YOLO aux branch | Optional cross-view fusion ablation | After P1 metrics |
| E6 — Oracle GT tracks vs predicted tracks | Quantify detector/tracker error propagation | P2–P3 diagnostic |
| E7 — Constant velocity vs learned prediction | Quantify temporal learning benefit | P3 required |
| E8 — Objects only vs +Occupancy risk | Quantify planning-relevant benefit | P4 required |

**Promotion criteria for optional YOLO/VRU head:** report paired per-scene VRU recall and precision changes, duplicate/fusion error cases, calibrated localization quality and end-to-end latency. No automatic promotion based on isolated illustrative clips. Numeric operating targets (e.g. recall or latency ceiling) must be set from a real operating design envelope, measured hardware and target safety scenario—not invented in this document.

## 10. Security, safety and research integrity

- All outputs are **research-grade**, not sufficient to claim ISO 26262/SOTIF compliance, full safety assurance, or deployment readiness.
- Dataset usage must follow official licensing and distribution constraints. Never commit or redistribute non-redistributable sensor data.
- Test-time future leakage, training on validation/test labels, fabricated ground truth, fabricated performance and fabricated checkpoint claims are prohibited.
- Calibration failures, absent depth, missing tracks, lack of label coverage and model uncertainty must be **explicit** in outputs and reports.
- Record dataset-domain shift (BDD100K → nuScenes, PIE → nuScenes, nuScenes → Waymo) and never claim cross-domain performance without real target-domain evaluation.
- Separate offline recorded-data evaluation from simulated closed-loop results. Neither demonstrates safe behavior in real traffic.
- Use pre-commit secret scanning or equivalent and commit no access tokens, personal paths or model-service keys.

## 11. Milestones / dependency summary

| Phase | Goal | Main output | Must pass before next phase |
|---|---|---|---|
| **1 — Unified 3D Detection** | BEVFusion detects vehicles + VRUs in 3D; YOLO separate baseline | Metric 3D boxes + validated nuScenes evaluations | Geometry/calibration tests + real fusion inference + official evaluation |
| **2 — 3D MOT** | Stable IDs, metric motion and histories | `TrackedAgent3D` + online MOT metrics | Ego-motion & scene-reset tests + real online track evaluation |
| **3 — Behavior & Prediction** | Crossing/action where labeled + future agent trajectories | Calibrated, validity-aware per-agent forecasts | No leakage, correct labels/splits, real trajectory and intention reports |
| **4 — Occupancy & Planning** | Semantic space understanding, collision checks, simulated ego motion | Occupancy + risk engine + replayable sim-only planner | Correct occupancy GT, scenario tests, safety guard + measured limitations |

## 12. Immediate instructions to the AI agent — START HERE

**Do not start Phase 2–4 implementation before building and validating the Phase 1 vertical slice.** The next work session must:

1. Read this file and inspect repository, assets, checkpoints and available GPU/dataset paths.
2. Create `docs/environment.md`, `docs/architecture.md`, `docs/progress.md`, `.gitignore`, `.env.example` and a minimal installable Python package.
3. Choose and **verify** the MMDetection3D BEVFusion codebase/config/checkpoint compatibility with local CUDA; record source tag/commit and exact commands.
4. Build a nuScenes-mini camera + LiDAR loader and a 3D-box/image projection sanity visualization.
5. Implement `Detection3D` contracts and one model adapter.
6. Execute actual BEVFusion inference on 1–2 real synchronized samples **if dataset/checkpoint available**; otherwise report the precise blocker and show only unit-test evidence.
7. Write CPU-only calibration/schema smoke tests.
8. Deliver a concise progress summary: modified files, how to run, commands actually executed, results/metrics, blockers and next action.

### Expected agent final response format after each work session

```text
Phase: <1|2|3|4>
Status: <IN_PROGRESS|BLOCKED|DONE>
Implemented: <specific files and behaviors>
Executed: <exact shell commands and results>
Evidence: <metrics/tests/screenshots paths; NOT_MEASURED if not run>
Blockers: <concrete missing inputs or incompatibilities>
Next: <one or two precisely scoped engineering tasks>
```

## 13. Official references / implementation sources

- [BEVFusion original research implementation](https://github.com/mit-han-lab/bevfusion) — original repository; archived, legacy dependencies.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) — actively maintainable integration candidate, inspect `projects/BEVFusion` for available configs and compatibility.
- [nuScenes developer kit](https://github.com/nutonomy/nuscenes-devkit) — dataset loading, map APIs and evaluation.
- [nuScenes tracking benchmark](https://www.nuscenes.org/tracking) — official 3D MOT rules/metrics.
- [nuScenes prediction benchmark](https://www.nuscenes.org/prediction) — official prediction task information.
- [PIE pedestrian intention dataset](https://github.com/aras62/PIE) — video/annotation resources and behavior labels.
- [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) — semantic 3D occupancy benchmark/labels.
- [TransFusion](https://arxiv.org/abs/2203.11496) — 3D detection with LiDAR-camera transformer fusion; related, not identical to BEVFusion.

**Reminder to future agents:** external repositories and documentation change. Verify exact file paths, weights, licenses, Python/CUDA requirements and published metric protocols *at execution time*. This file is the objective contract, not proof that dependencies or artifacts have already been downloaded.
