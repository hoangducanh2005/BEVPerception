#!/usr/bin/env python3
"""Run a small Phase 1 integration smoke without downloading data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phase1_common import load_yaml, manifest_base, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    runtime = config.setdefault("runtime", {})
    seed = args.seed if args.seed is not None else int(runtime.get("seed", 0))
    set_seed(seed)
    if args.device:
        runtime["device"] = args.device
    limit = args.limit_samples if args.limit_samples is not None else int(runtime.get("limit_samples", 2))
    manifest = manifest_base(args.config, config, sys.argv)
    manifest.update({"command_name": "phase1_smoke", "limit_samples": limit, "seed": seed})
    output_dir = args.output_dir
    try:
        from bevdrive.datasets.nuscenes_adapter import NuScenesAdapter

        root = config["dataset"]["root"]
        if not root or root.startswith("${"):
            raise RuntimeError("NUSCENES_ROOT is not set; refusing to use a guessed dataset path")
        adapter = NuScenesAdapter(
            root,
            config["dataset"].get("version", "v1.0-mini"),
            split=config["dataset"].get("split", "mini_val"),
            max_timestamp_delta_ms=float(config["dataset"].get("max_timestamp_delta_ms", 100)),
        )
        frames = list(adapter.iter_frames(limit))
        if len(frames) < 2:
            raise RuntimeError(f"Need at least 2 real samples for the Phase 1 gate, found {len(frames)}")
        manifest.update({"status": "DATA_READY_MODEL_BLOCKED", "num_frames": len(frames)})
        manifest["reason"] = (
            "Dataset metadata is readable. Model inference remains pending until a checkpoint is configured "
            "and BEVFusion CUDA extensions are built."
        )
        manifest["samples"] = [frame.to_dict() for frame in frames]
    except Exception as exc:  # CLI must record an actionable blocked state.
        manifest.update({"status": "BLOCKED", "reason": str(exc)})
    write_json(output_dir / "smoke_manifest.json", manifest)
    print(f"Phase 1 smoke: {manifest['status']}")
    print(f"Manifest: {output_dir / 'smoke_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
