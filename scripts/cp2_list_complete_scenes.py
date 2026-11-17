#!/usr/bin/env python3
"""List nuScenes scenes whose keyframes (6 cameras + LiDAR) and 9 LiDAR sweeps all exist on disk.

The local trainval copy is partial (one blob), so only complete scenes can be
used. Output: JSON {"train": [...], "val": [...], "frames": {...}} by official split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import scene_is_complete  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/media/anhhd/New Volume/nuscenes"))
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--out", type=Path, default=REPO / "artifacts/checkpoint_2/trainval_complete_scenes.json")
    args = p.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    splits = create_splits_scenes()
    lidar_files = {f.name for f in (args.data_root / "samples/LIDAR_TOP").iterdir()}
    out = {"train": [], "val": [], "frames": {}}
    for scene in nusc.scene:
        first = nusc.get("sample", scene["first_sample_token"])
        if Path(nusc.get("sample_data", first["data"]["LIDAR_TOP"])["filename"]).name not in lidar_files:
            continue
        if not scene_is_complete(nusc, args.data_root, scene["name"]):
            continue
        split = "train" if scene["name"] in splits["train"] else "val"
        out[split].append(scene["name"])
        out["frames"][scene["name"]] = scene["nbr_samples"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"complete scenes: train {len(out['train'])} ({sum(out['frames'][s] for s in out['train'])} frames), "
          f"val {len(out['val'])} ({sum(out['frames'][s] for s in out['val'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
