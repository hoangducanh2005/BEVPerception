# Phase 1 plan — Unified 3D Detection

Status: `IN_PROGRESS`; the local runtime and project CUDA extensions are
verified, but the real-data/checkpoint gate is still open.

## Required work

1. Build a scene-safe nuScenes adapter and validate sensor metadata,
   calibration, timestamp alignment, and camera/LiDAR projections.
2. Add geometry projection/transform functions and a typed `Detection3D`
   contract with metric units and explicit coordinate frames.
3. Establish a LiDAR-only 3D baseline and the camera+LiDAR BEVFusion detector
   on the same nuScenes split.
4. Keep YOLO as an independent 2D camera baseline; do not compare its 2D mAP
   numerically with 3D metrics.
5. Produce calibration overlays, per-class vehicle/VRU metrics, error galleries,
   run manifests, and latency/memory evidence.
6. Only consider fine-tuning after baseline evaluation and compatible resources
   have been confirmed.

## Environment and inputs to resolve first

- MMDetection3D checkout: `/home/anhhd/GIT/mmdetection3d`, branch `dev-1.x`,
  commit `962f093736ffe55c089bc618842a8b8567318c8c`.
- Verified local matrix: Python 3.10.22, PyTorch 2.1.0+cu121, MMCV 2.1.0
  source-built with CUDA ops, MMEngine 0.10.7, MMDetection 3.2.0, and
  MMDetection3D 1.4.0. `nvcc` is 12.0 and the GPU is an RTX 4050 Laptop.
- The OpenMMLab wheel host was unavailable during setup, so the reproducible
  fallback is the source-build command recorded in `docs/environment.md`.
- nuScenes v1.0-mini for smoke validation; trainval for meaningful evaluation
  when available and licensed. No dataset was found in the initial local scan.
- Compatible BEVFusion and LiDAR checkpoint metadata/weights; none were found
  in the initial local scan.
- BDD100K YOLO checkpoint is optional and independent; none was found in the
  initial local scan.
- Local machine has 6 GiB GPU VRAM and 15 GiB RAM. Treat training capacity as
  unverified; begin with low-memory inference/smoke settings. `open3d` is a
  remaining heavy install, and no dataset/checkpoint has been provisioned yet.

## Gate

Do not mark Phase 1 done until at least two real synchronized nuScenes samples
produce valid BEVFusion outputs, geometry tests pass, LiDAR-only and fusion
inference are run or explicitly blocked, and a legitimate nuScenes evaluator
run, visual review, and latency result (or `NOT_MEASURED`) are recorded.
