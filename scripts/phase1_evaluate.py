#!/usr/bin/env python3
"""Prepare a reproducible Phase 1 evaluation run or report its blocker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phase1_common import load_yaml, manifest_base, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    runtime = config.setdefault("runtime", {})
    seed = args.seed if args.seed is not None else int(runtime.get("seed", 0))
    set_seed(seed)
    if args.device:
        runtime["device"] = args.device
    report = manifest_base(args.config, config, sys.argv)
    report.update({"command_name": "phase1_evaluate", "split": args.split, "seed": seed, "metrics": "NOT_MEASURED"})
    try:
        from bevdrive.datasets.nuscenes_adapter import NuScenesAdapter

        root = config["dataset"]["root"]
        if not root or root.startswith("${"):
            raise RuntimeError("NUSCENES_ROOT is not set; official evaluation cannot run")
        adapter = NuScenesAdapter(
            root,
            config["dataset"].get("version", "v1.0-mini"),
            split=config["dataset"].get("split", "mini_val"),
            max_timestamp_delta_ms=float(config["dataset"].get("max_timestamp_delta_ms", 100)),
        )
        limit = args.limit_samples or int(runtime.get("limit_samples", 2))
        frames = list(adapter.iter_frames(limit))
        if not frames:
            raise RuntimeError("No samples found for the requested split")
        checkpoint = config["paths"].get("checkpoint", "")
        if not checkpoint or checkpoint.startswith("${"):
            raise RuntimeError("BEVFUSION_CHECKPOINT is not set; model evaluation is blocked")
        report.update({"status": "BLOCKED", "num_frames": len(frames), "reason": "Model inference adapter is not yet executed"})
    except Exception as exc:
        report.update({"status": "BLOCKED", "reason": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", report)
    write_json(args.output_dir / "run_manifest.json", report)
    (args.output_dir / "summary.md").write_text(
        f"# Phase 1 evaluation\n\nStatus: `{report['status']}`\n\n"
        f"Reason: {report.get('reason', 'not provided')}\n\nMetrics: `NOT_MEASURED` until real model inference and the nuScenes evaluator run.\n",
        encoding="utf-8",
    )
    print(f"Phase 1 evaluation: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
