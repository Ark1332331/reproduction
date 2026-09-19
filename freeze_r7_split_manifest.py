"""Freeze explicit train/validation NPZ membership for R7 experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TERRAINS = ("stairs", "boxes", "walls", "poles", "corridors")


def accepted(path: Path, min_frames: int) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as payload:
        frames = sum(name.startswith("measurement_") for name in payload.files)
        metadata = json.loads(str(payload["metadata_json"]))
    return frames, int(metadata["scene_seed"])


def collect(directory: Path, min_frames: int) -> dict[str, dict]:
    result = {}
    for terrain in TERRAINS:
        files = []
        seeds = []
        frames = 0
        for path in sorted(directory.glob(f"isaac_anymal_{terrain}_*.npz")):
            count, seed = accepted(path, min_frames)
            if count < min_frames:
                continue
            files.append(str(path.resolve()))
            seeds.append(seed)
            frames += count
        result[terrain] = {
            "files": files,
            "accepted_trajectories": len(files),
            "accepted_frames": frames,
            "scene_seeds": sorted(set(seeds)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-trajectory-frames", type=int, default=10)
    args = parser.parse_args()
    train = collect(args.train_dir, args.min_trajectory_frames)
    validation = collect(args.validation_dir, args.min_trajectory_frames)
    overlaps = {
        terrain: sorted(set(train[terrain]["scene_seeds"]) & set(validation[terrain]["scene_seeds"]))
        for terrain in TERRAINS
    }
    if any(overlaps.values()):
        raise RuntimeError(f"train/validation scene-seed overlap: {overlaps}")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_trajectory_frames": args.min_trajectory_frames,
        "train_dir": str(args.train_dir.resolve()),
        "validation_dir": str(args.validation_dir.resolve()),
        "train": train,
        "validation": validation,
        "scene_seed_overlaps": overlaps,
        "frozen": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "frozen": True,
        "train_frames": sum(x["accepted_frames"] for x in train.values()),
        "validation_frames": sum(x["accepted_frames"] for x in validation.values()),
        "validation_trajectories": {t: validation[t]["accepted_trajectories"] for t in TERRAINS},
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
