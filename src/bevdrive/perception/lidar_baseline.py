"""LiDAR-only MMDetection3D baseline adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bev_detector import BEVFusionDetector


@dataclass(frozen=True, slots=True)
class LidarOnlyConfig:
    config_path: Path
    checkpoint_path: Path | None
    mmdetection3d_root: Path
    device: str = "cuda:0"


class LidarOnlyDetector(BEVFusionDetector):
    """Use the same stable wrapper boundary for a LiDAR-only config."""

    def __init__(self, config: LidarOnlyConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]

    def predict(self, sample: Mapping[str, Any]) -> Any:
        return super().predict(sample)
