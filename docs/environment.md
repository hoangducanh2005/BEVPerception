# Environment audit

Audit date: 2026-10-22 (Asia/Ho_Chi_Minh)

## Observed machine

| Item | Observed value |
|---|---|
| OS/kernel | Ubuntu 24.04 family, x86_64; kernel `7.0.0-38-generic` |
| Python | System `python3` is 3.12.3; `python` command is absent |
| pip | Not installed for system Python (`python3 -m pip` reports no module) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB reported VRAM |
| NVIDIA driver | 580.178.04; `nvidia-smi` reports CUDA 13.0 driver capability |
| CUDA compiler toolkit | `nvcc` 12.0.140 |
| RAM / swap | 15 GiB / 4 GiB |
| Repository filesystem | 468 GiB total, 413 GiB free at audit time |
| Conda | Miniconda present; environments `base`, `ai_core` |

Driver-reported CUDA capability and the installed CUDA compiler toolkit are
different facts. Neither alone selects the PyTorch wheel or proves compatibility
with MMCV CUDA operators. Select and verify one complete PyTorch/CUDA/MMCV/
MMDetection/MMDetection3D matrix using the chosen BEVFusion project revision
before locking dependencies. Do not install CUDA system-wide for this project.

## Existing files and local assets

The BEVPerception repository initially contained only `objectives.md` and
`prediction/README.txt` before setup scaffolding was added. No nuScenes,
BDD100K, or model checkpoint was found in the repository or by a shallow search
of `/home/anhhd`. This does not rule out data mounted elsewhere.

The existing `ai_core` Conda environment now has the verified Phase 1 matrix:

| Package/tool | Version or source |
|---|---|
| Python | 3.10.22 |
| PyTorch / torchvision / torchaudio | 2.1.0+cu121 / 0.16.0+cu121 / 2.1.0+cu121 |
| CUDA visible to PyTorch | 12.1; `torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| MMCV | 2.1.0, source-built with CUDA ops |
| MMEngine / MMDetection / MMDetection3D | 0.10.7 / 3.2.0 / 1.4.0 |
| nuScenes devkit / numba | 1.2.0 / 0.59.1 |

The system compiler is `nvcc` 12.0 while the PyTorch runtime is cu121. The
MMCV and BEVFusion extension builds completed with the expected minor-version
warning, and their CUDA smoke tests passed.

The official MMDetection3D checkout has been downloaded to
`/home/anhhd/GIT/mmdetection3d`, branch `dev-1.x`, commit
`962f093736ffe55c089bc618842a8b8567318c8c` (shallow clone). The same checkout
contains the BEVFusion project and its README/configs. It is installed
editable and its CUDA extensions have been built successfully. The official
OpenMMLab wheel host timed out, so MMCV 2.1.0 was built from source with:

```bash
CUDA_HOME=/usr/lib/nvidia-cuda-toolkit \
MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
python -m pip install --no-build-isolation --no-deps --no-binary=mmcv mmcv==2.1.0
```

`nvidia-smi`, `nvcc --version`, `free -h`, `df -h .`, and the existing Conda
environment/package checks were run.

## Environment setup decision

Use the existing `ai_core` Conda environment rather than the system Python.
The core matrix and project extensions are installed and verified. The heavy
`open3d` wheel (~400 MB) was not completed because its download stalled at
about 148 MB; it must be installed before using MMDetection3D features that
require it.

The RTX 4050 Laptop's 6 GiB VRAM and system RAM are constrained for camera-LiDAR
fusion. Start with inference/smoke work at a low sample count and batch size;
measure peak memory before attempting training. Do not claim training feasibility
until the selected config has been checked against available memory.

## Local configuration

Copy `.env.example` to `.env` and set local dataset/checkpoint paths. `.env`,
data, checkpoints, outputs, and run artifacts are gitignored. Do not put absolute
machine-specific paths or credentials in tracked configs.

## Verification after manual MMDetection3D installation

The following checks passed inside `ai_core`; they do not download a dataset or
checkpoint:

```bash
which python
python --version
python -c "import torch; print('torch', torch.__version__); print('torch CUDA', torch.version.cuda); print('CUDA available', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import mmengine, mmcv, mmdet, mmdet3d; print('mmengine', mmengine.__version__); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmdet3d', mmdet3d.__version__)"
```

With `PYTHONPATH=/home/anhhd/GIT/mmdetection3d/projects/BEVFusion`, the
`bevfusion` package imports, voxelization returns CUDA results, and BEV pooling
returns a CUDA tensor. Repository tests report `7 passed` and Ruff reports
`All checks passed`.

No nuScenes dataset or BEVFusion/LiDAR checkpoint is present yet. Therefore no
real-sample inference, official evaluation, latency measurement, or Phase 1
acceptance decision has been made.
