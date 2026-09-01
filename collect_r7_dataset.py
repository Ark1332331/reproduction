"""Run recoverable batches of the existing IsaacLab ANYmal trajectory collector.

This launcher does not alter the scene or invent extra samples.  It turns the
single-trajectory collector into an auditable batch: every terrain/seed pair
gets one output path, one log, and one manifest record.  The manifest is
rewritten after each job, so a later invocation can skip completed captures and
resume the remainder.

Run from the project root, preferably first with ``--dry-run``:

    python reproduction/collect_r7_dataset.py --split train-extra \
        --seed-start 4 --seed-count 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_TERRAINS = ("stairs", "boxes", "walls", "poles", "corridors")


@dataclass(frozen=True)
class CaptureJob:
    """One independent terrain layout and one robot trajectory capture."""

    terrain: str
    seed: int
    output: Path
    log: Path


def build_jobs(
    data_dir: Path,
    log_dir: Path,
    terrains: tuple[str, ...],
    seed_start: int,
    seed_count: int,
) -> tuple[CaptureJob, ...]:
    """Give every requested terrain/seed pair a stable, non-overlapping name."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    jobs = []
    for seed in range(seed_start, seed_start + seed_count):
        for terrain in terrains:
            stem = f"isaac_anymal_{terrain}_s{seed}"
            jobs.append(CaptureJob(terrain, seed, data_dir / f"{stem}.npz", log_dir / f"{stem}.log"))
    return tuple(jobs)


def capture_summary(path: Path) -> dict | None:
    """Return the minimal R7 file-contract evidence, or ``None`` when invalid."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as capture:
            metadata = json.loads(str(capture["metadata_json"].item()))
            frame_count = metadata.get("trajectory_steps")
            if metadata.get("coordinate_frame") != "robot_centric_local_map" or not isinstance(frame_count, int) or frame_count <= 1:
                return None
            for index in range(frame_count):
                suffix = f"{index:02d}"
                required = (f"measurement_{suffix}", f"target_{suffix}", f"translation_{suffix}", f"yaw_{suffix}")
                if any(name not in capture for name in required):
                    return None
            return {
                "captured_frames": frame_count,
                "terrain": metadata.get("terrain"),
                "scene_seed": metadata.get("scene_seed"),
            }
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def completed_capture(path: Path) -> bool:
    """A completed capture must satisfy the basic R7 NPZ contract, not just exist."""
    return capture_summary(path) is not None


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def command_for(
    isaaclab_root: Path,
    collector: Path,
    job: CaptureJob,
    trajectory_steps: int,
    points_per_camera: int,
    isaaclab_env: str = "isaaclab",
) -> list[str]:
    # IsaacLab launches the child Python process from its own installation root.
    # Relative project paths would therefore resolve under ``IsaacLab/`` rather
    # than this workspace; pass the capture contract across that boundary using
    # absolute paths.
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        isaaclab_env,
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(collector.resolve()),
        "--headless",
        "--enable_cameras",
        "--terrain",
        job.terrain,
        "--seed",
        str(job.seed),
        "--trajectory-steps",
        str(trajectory_steps),
        "--points-per-camera",
        str(points_per_camera),
        "--output",
        str(job.output.resolve()),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recoverable batch launcher for R7 IsaacLab captures.")
    parser.add_argument("--data-dir", type=Path, default=Path("reproduction/data"))
    parser.add_argument("--log-dir", type=Path, default=Path("reproduction/data/collection_logs"))
    parser.add_argument("--manifest", type=Path, default=Path("reproduction/data/r7_collection_manifest.json"))
    parser.add_argument("--split", required=True, help="Human-readable label recorded in the manifest.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--terrains", nargs="+", choices=DEFAULT_TERRAINS, default=DEFAULT_TERRAINS)
    parser.add_argument("--trajectory-steps", type=int, default=12)
    parser.add_argument("--points-per-camera", type=int, default=1500)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/ark/projects/IsaacLab"))
    parser.add_argument("--isaaclab-env", default="isaaclab", help="Conda environment containing IsaacLab.")
    parser.add_argument("--collector", type=Path, default=Path("reproduction/collect_isaaclab_anymal_trajectory.py"))
    parser.add_argument("--overwrite", action="store_true", help="Recapture even when a nonempty NPZ already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned manifest without launching IsaacLab.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with later jobs after one capture failure.")
    args = parser.parse_args()

    if args.trajectory_steps <= 1 or args.points_per_camera <= 0:
        raise ValueError("trajectory_steps must exceed one and points_per_camera must be positive")
    launcher = args.isaaclab_root / "isaaclab.sh"
    if not args.dry_run and not launcher.is_file():
        raise FileNotFoundError(f"IsaacLab launcher not found: {launcher}")
    if not args.collector.is_file():
        raise FileNotFoundError(f"collector not found: {args.collector}")

    jobs = build_jobs(args.data_dir, args.log_dir, tuple(args.terrains), args.seed_start, args.seed_count)
    manifest = {
        "purpose": "serial, recoverable IsaacLab R7 data collection; not a claim of paper-scale data",
        "created_at": now(),
        "split": args.split,
        "trajectory_steps_requested": args.trajectory_steps,
        "points_per_camera": args.points_per_camera,
        "jobs": [],
    }
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        record = {"job": asdict(job), "status": "planned"}
        record["job"]["output"] = str(job.output)
        record["job"]["log"] = str(job.log)
        if completed_capture(job.output) and not args.overwrite:
            record.update({"status": "skipped_existing", "finished_at": now(), **capture_summary(job.output)})
            print(f"[SKIP] {job.output}", flush=True)
        else:
            command = command_for(
                args.isaaclab_root,
                args.collector,
                job,
                args.trajectory_steps,
                args.points_per_camera,
                args.isaaclab_env,
            )
            record["command"] = command
            if args.dry_run:
                record.update({"status": "dry_run", "finished_at": now()})
                print("[DRY] " + " ".join(command), flush=True)
            else:
                record["started_at"] = now()
                started = time.monotonic()
                # IsaacLab's shell launcher runs terminal setup commands; in a
                # non-interactive subprocess Codex may provide TERM=dumb, which
                # makes `tabs` fail before Python/IsaacLab even starts.
                environment = dict(os.environ, TERM="xterm")
                with job.log.open("w") as log_file:
                    result = subprocess.run(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=environment,
                    )
                record.update(
                    {
                        "status": "completed" if result.returncode == 0 and completed_capture(job.output) else "failed",
                        "returncode": result.returncode,
                        "duration_seconds": round(time.monotonic() - started, 1),
                        "finished_at": now(),
                    }
                )
                if record["status"] == "completed":
                    record.update(capture_summary(job.output))
                print(f"[{record['status'].upper()}] {job.terrain} seed={job.seed}", flush=True)
        manifest["jobs"].append(record)
        write_manifest(args.manifest, manifest)
        if record["status"] == "failed" and not args.keep_going:
            print(f"[STOP] see {job.log}", file=sys.stderr, flush=True)
            return 1

    print(f"[DONE] {len(jobs)} jobs recorded in {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
