"""Shared CLI/config helpers for Phase 1 commands."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 1 configuration files") from exc
    config_path = Path(path).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return _expand_values(values)


def _expand_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manifest_base(config_path: Path, config: dict[str, Any], command: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command": command,
        "config": str(config_path.resolve()),
        "config_values": config,
        "repository": str(REPO_ROOT),
        "status": "NOT_MEASURED",
    }
    try:
        import torch
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["torch"] = None
    return result


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
