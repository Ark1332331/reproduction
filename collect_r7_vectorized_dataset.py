"""Recoverable launcher for batches of four parallel R7 IsaacLab trajectories.

The vectorized collector owns the simulator/data contract.  This module only
plans terrain/seed batches, gives each one a log and writes a manifest after
each attempt.  It therefore remains possible to stop after any batch and see
which individual trajectories are actually usable.

Example plan (does not start Isaac Sim):

    python reproduction/collect_r7_vectorized_dataset.py --split train-extra \\
        --seed-start 60 --batch-count 2 --dry-run
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

from collect_r7_dataset import capture_summary
from r7_vectorized_capture_contract import CAMERA_DIRECTIONS, output_path_for_environment

DEFAULT_TERRAINS = ("stairs", "boxes", "walls", "poles", "corridors")


@dataclass(frozen=True)
class VectorizedCaptureJob:
    """One simulator launch containing ``num_envs`` terrain-layout trajectories."""

    terrain: str
    base_seed: int
    output_dir: Path
    log: Path


def build_jobs(
    data_dir: Path,
    log_dir: Path,
    terrains: tuple[str, ...],
    seed_start: int,
    batch_count: int,
    num_envs: int,
) -> tuple[VectorizedCaptureJob, ...]:
    """Use non-overlapping seed ranges: batch j owns [seed, seed + num_envs)."""
    if seed_start < 0 or batch_count <= 0 or num_envs <= 0:
        raise ValueError("seed_start must be non-negative; batch_count and num_envs must be positive")
    jobs = []
    for batch_index in range(batch_count):
        base_seed = seed_start + batch_index * num_envs
        for terrain in terrains:
            stem = f"vectorized_{terrain}_s{base_seed}_n{num_envs}"
            jobs.append(VectorizedCaptureJob(terrain, base_seed, data_dir, log_dir / f"{stem}.log"))
    return tuple(jobs)


def job_outputs(job: VectorizedCaptureJob, num_envs: int) -> tuple[Path, ...]:
    return tuple(output_path_for_environment(job.output_dir, job.terrain, job.base_seed, index) for index in range(num_envs))


def completed_job(job: VectorizedCaptureJob, num_envs: int) -> bool:
    """A batch is complete only when every environment made a valid R7 sample."""
    return all(capture_summary(path) is not None for path in job_outputs(job, num_envs))


def command_for(
    isaaclab_root: Path,
    collector: Path,
    job: VectorizedCaptureJob,
    num_envs: int,
    trajectory_steps: int,
    points_per_camera: int,
    isaaclab_env: str,
) -> list[str]:
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
        str(job.base_seed),
        "--num-envs",
        str(num_envs),
        "--trajectory-steps",
        str(trajectory_steps),
        "--points-per-camera",
        str(points_per_camera),
        "--output-dir",
        str(job.output_dir.resolve()),
    ]


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Recoverable vectorized R7 IsaacLab collection launcher.")
    parser.add_argument("--data-dir", type=Path, default=Path("reproduction/data/vectorized_train_extra"))
    parser.add_argument("--log-dir", type=Path, default=Path("reproduction/data/collection_logs_vectorized"))
    parser.add_argument("--manifest", type=Path, default=Path("reproduction/data/r7_vectorized_collection_manifest.json"))
    parser.add_argument("--split", required=True, help="Human-readable label; not a training/validation claim.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--batch-count", type=int, required=True, help="Each batch starts one simulator with num-envs trajectories.")
    parser.add_argument("--num-envs", type=int, default=4, help="Verified default for this workstation.")
    parser.add_argument("--terrains", nargs="+", choices=DEFAULT_TERRAINS, default=DEFAULT_TERRAINS)
    parser.add_argument("--trajectory-steps", type=int, default=12)
    parser.add_argument("--points-per-camera", type=int, default=1500)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/ark/projects/IsaacLab"))
    parser.add_argument("--isaaclab-env", default="isaaclab")
    parser.add_argument("--collector", type=Path, default=Path("reproduction/collect_isaaclab_anymal_vectorized.py"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.trajectory_steps <= 1 or args.points_per_camera <= 0:
        raise ValueError("trajectory-steps must exceed one and points-per-camera must be positive")
    if not args.dry_run and not (args.isaaclab_root / "isaaclab.sh").is_file():
        raise FileNotFoundError(f"IsaacLab launcher not found: {args.isaaclab_root / 'isaaclab.sh'}")
    if not args.collector.is_file():
        raise FileNotFoundError(f"vectorized collector not found: {args.collector}")

    jobs = build_jobs(args.data_dir, args.log_dir, tuple(args.terrains), args.seed_start, args.batch_count, args.num_envs)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "recoverable vectorized R7 collection; not a paper-scale or training-result claim",
        "created_at": now(),
        "split": args.split,
        "num_envs": args.num_envs,
        "camera_count_per_environment": len(CAMERA_DIRECTIONS),
        "trajectory_steps_requested": args.trajectory_steps,
        "points_per_camera": args.points_per_camera,
        "jobs": [],
    }

    for job in jobs:
        record = {"job": asdict(job), "outputs": [str(path) for path in job_outputs(job, args.num_envs)], "status": "planned"}
        record["job"]["output_dir"] = str(job.output_dir)
        record["job"]["log"] = str(job.log)
        if completed_job(job, args.num_envs) and not args.overwrite:
            record.update(
                {
                    "status": "skipped_existing",
                    "finished_at": now(),
                    "captures": [capture_summary(path) for path in job_outputs(job, args.num_envs)],
                }
            )
            print(f"[SKIP] {job.terrain} seed={job.base_seed}", flush=True)
        else:
            command = command_for(
                args.isaaclab_root, args.collector, job, args.num_envs, args.trajectory_steps, args.points_per_camera, args.isaaclab_env
            )
            record["command"] = command
            if args.dry_run:
                record.update({"status": "dry_run", "finished_at": now()})
                print("[DRY] " + " ".join(command), flush=True)
            else:
                record["started_at"] = now()
                started = time.monotonic()
                environment = dict(os.environ, TERM="xterm")
                with job.log.open("w") as log_file:
                    result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False, env=environment)
                captures = [capture_summary(path) for path in job_outputs(job, args.num_envs)]
                record.update(
                    {
                        "status": "completed" if result.returncode == 0 and all(captures) else "failed",
                        "returncode": result.returncode,
                        "duration_seconds": round(time.monotonic() - started, 1),
                        "finished_at": now(),
                        "captures": captures,
                    }
                )
                print(f"[{record['status'].upper()}] {job.terrain} seed={job.base_seed}", flush=True)
        manifest["jobs"].append(record)
        write_manifest(args.manifest, manifest)
        if record["status"] == "failed" and not args.keep_going:
            print(f"[STOP] see {job.log}", file=sys.stderr, flush=True)
            return 1

    print(f"[DONE] {len(jobs)} vectorized batches recorded in {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
