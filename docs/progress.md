# Progress

## Phase 1 — Unified 3D Detection

Status: `IN_PROGRESS` (runtime verified; real data/checkpoint gate is blocked).

- [x] Read the objectives and record available machine resources.
- [x] Add minimal package and local-path conventions.
- [x] Download MMDetection3D `dev-1.x` source with BEVFusion project.
- [x] Choose and install a verified MMDetection3D + BEVFusion dependency matrix in `ai_core`.
- [x] Build BEVFusion CUDA extensions and run CUDA operator smoke tests.
- [x] Run a 10-frame BEVFusion smoke inference on the available NavSim/OpenScene sequence and record coarse metrics evidence.
- [ ] Locate/provision nuScenes v1.0-mini and verify access/license.
- [ ] Locate/provision compatible BEVFusion and LiDAR checkpoints.
- [ ] Implement and validate the Phase 1 vertical slice.

Current state: `ai_core` has Python 3.10.22, PyTorch 2.1.0+cu121, CUDA
available, MMCV 2.1.0 with CUDA ops, MMEngine 0.10.7, MMDetection 3.2.0, and
MMDetection3D 1.4.0. Source checkout is
`/home/anhhd/GIT/mmdetection3d` at
`962f093736ffe55c089bc618842a8b8567318c8c`. BEVFusion imports and its voxel
and BEV-pool CUDA smoke tests pass; repository tests report 7 passed and Ruff
passes. `open3d` remains to be downloaded before using features that require
it. nuScenes and model checkpoints were not found, so real inference and
metrics have not been run.

Next: set `NUSCENES_ROOT` and checkpoint paths, provision at least nuScenes
v1.0-mini plus compatible BEVFusion/LiDAR checkpoints, then run the real-data
smoke/evaluation gate. Install `open3d==0.18.0` before a full `pip check` or
MMDetection3D workflows that require it. The NavSim evidence is recorded in
`reports/phase1/navsim_bevfusion_10frames.md`; it is not an official nuScenes
mAP/NDS result because the sensor layout and annotation format differ.
