"""Optional camera-only YOLO baseline.

YOLO is intentionally isolated from the BEVFusion inference path.  This
adapter is only activated when a checkpoint and matching image list are
explicitly configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Yolo2DConfig:
    checkpoint_path: Path
    device: str = "cuda:0"


class Yolo2DDetector:
    def __init__(self, config: Yolo2DConfig) -> None:
        self.config = config
        self.model: Any | None = None

    def load(self) -> Any:
        if not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.config.checkpoint_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed; YOLO is an optional Phase 1 baseline") from exc
        self.model = YOLO(str(self.config.checkpoint_path))
        return self.model

    def predict(self, image_path: str | Path) -> Any:
        if self.model is None:
            self.load()
        return self.model.predict(source=str(image_path), device=self.config.device, verbose=False)
