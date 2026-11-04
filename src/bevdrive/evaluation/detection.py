"""Small, deterministic report utilities for Phase 1 artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from bevdrive.contracts.detection import Detection3D


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def build_detection_report(detections: Iterable[Detection3D], *, status: str, reason: str | None = None) -> dict[str, Any]:
    values = list(detections)
    report: dict[str, Any] = {
        "status": status,
        "num_detections": len(values),
        "class_counts": dict(sorted(Counter(item.class_name for item in values).items())),
        "scores": [item.score for item in values],
        "coordinate_frames": sorted({item.coordinate_frame for item in values}),
        "source": sorted({item.source for item in values}),
    }
    if reason:
        report["reason"] = reason
    return report
