#!/usr/bin/env python3
"""Guarded entry point for a resumable Phase 1 training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phase1_common import load_yaml, manifest_base, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", default="auto")
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
    report = manifest_base(args.config, config, sys.argv)
    report.update({"command_name": "phase1_train", "resume": args.resume, "seed": seed, "status": "BLOCKED"})
    report["reason"] = (
        "Training is intentionally not started: Phase 1 requires verified real-data baseline inference, "
        "checkpoint/config compatibility and a memory budget on the 6 GiB GPU."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_manifest.json", report)
    print("Phase 1 training: BLOCKED")
    print(report["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
