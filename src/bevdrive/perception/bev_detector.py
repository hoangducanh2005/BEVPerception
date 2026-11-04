"""Adapter around the MMDetection3D BEVFusion project.

This module does not vendor MMDetection3D.  The external checkout and its
compiled CUDA operations are configured explicitly, so a missing installation
fails with an actionable message.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BEVFusionConfig:
    config_path: Path
    checkpoint_path: Path | None
    mmdetection3d_root: Path
    device: str = "cuda:0"


class BEVFusionDetector:
    """Lazy MMDetection3D model wrapper with a stable repository boundary."""

    def __init__(self, config: BEVFusionConfig) -> None:
        self.config = config
        self.model: Any | None = None

    def load(self) -> Any:
        """Build the configured model and import the BEVFusion project."""

        if not self.config.config_path.is_file():
            raise FileNotFoundError(f"BEVFusion config not found: {self.config.config_path}")
        if self.config.checkpoint_path is not None and not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(f"BEVFusion checkpoint not found: {self.config.checkpoint_path}")
        root = str(self.config.mmdetection3d_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from mmdet3d.apis import init_model
            from projects.BEVFusion import bevfusion as _bevfusion  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MMDetection3D/BEVFusion is not importable. Install the locked dependencies and "
                "build projects/BEVFusion/setup.py in ai_core."
            ) from exc
        self.model = init_model(
            str(self.config.config_path),
            str(self.config.checkpoint_path) if self.config.checkpoint_path else None,
            device=self.config.device,
        )
        return self.model

    def predict(self, sample: Mapping[str, Any]) -> Any:
        """Run one prepared MMDetection3D sample.

        Data preparation remains explicit because nuScenes pipelines differ
        between MMDetection3D revisions.  The wrapper accepts either a sample
        mapping already prepared by the selected pipeline or raises a clear
        error instead of silently guessing field names.
        """

        if self.model is None:
            self.load()
        try:
            from mmdet3d.apis import inference_detector
        except ImportError as exc:
            raise RuntimeError("MMDetection3D inference API is unavailable") from exc
        return inference_detector(self.model, sample)
