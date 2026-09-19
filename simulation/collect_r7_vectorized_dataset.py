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
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulation.collect_r7_dataset import capture_summary
from simulation.r7_vectorized_capture_contract import CAMERA_DIRECTIONS, output_path_for_environment
from configs.paper_config import (
    PAPER_ROLLOUT_STEPS,
    PAPER_TERRAINS,
    REPRODUCTION_CAPTURE_SCHEMA_VERSION,
    REPRODUCTION_TERRAIN_PROFILE,
)

DEFAULT_TERRAINS = PAPER_TERRAINS

# These errors indicate a simulator/host failure, not a bad terrain seed.  A
# seed retry cannot repair them and would only create another concurrent GPU
# allocation if the failed Isaac Sim child escaped its wrapper.
SIMULATOR_FAILURE_PATTERNS = (
    ("cuda_oom", ("cuda error: out of memory", "out of memory", "failed to allocate memory")),
    ("device_lost", ("error_device_lost", "device lost", "gpu crash", "gpu pagefault")),
    ("segmentation_fault", ("segmentation fault", "sigsegv", "core dumped")),
)


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


def completed_job(job: VectorizedCaptureJob, num_envs: int, min_trajectory_frames: int = 2) -> bool:
    """A batch is complete only when every environment meets the frame gate and R7 contract."""
    summaries = [capture_summary(path) for path in job_outputs(job, num_envs)]
    return all(summary is not None and summary["captured_frames"] >= min_trajectory_frames for summary in summaries)


def command_for(
    isaaclab_root: Path,
    collector: Path,
    job: VectorizedCaptureJob,
    num_envs: int,
    trajectory_steps: int,
    points_per_camera: int,
    isaaclab_env: str,
    random_motion: bool = True,
    yaw_rate_min: float = -0.35,
    yaw_rate_max: float = 0.35,
    forward_velocity_min: float = 0.5,
    forward_velocity_max: float = 1.0,
    lateral_velocity_min: float = -0.2,
    lateral_velocity_max: float = 0.2,
) -> list[str]:
    command = [
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
    command.append("--random-motion" if random_motion else "--no-random-motion")
    if random_motion:
        command.extend([
            "--yaw-rate-min", str(yaw_rate_min), "--yaw-rate-max", str(yaw_rate_max),
            "--forward-velocity-min", str(forward_velocity_min), "--forward-velocity-max", str(forward_velocity_max),
            "--lateral-velocity-min", str(lateral_velocity_min), "--lateral-velocity-max", str(lateral_velocity_max),
        ])
    return command


def now() -> str:
    return datetime.now(UTC).isoformat()


def classify_log_failure(log: Path) -> str | None:
    """Classify host/simulator failures so they are not blindly resampled."""
    try:
        text = log.read_text(errors="replace").lower()
    except OSError:
        return None
    for kind, patterns in SIMULATOR_FAILURE_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return kind
    return None


def terminate_process_group(pgid: int, force: bool = False) -> None:
    """Stop a simulator attempt and all of its descendants."""
    try:
        os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Recoverable vectorized R7 IsaacLab collection launcher.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data/vectorized_train_extra")
    parser.add_argument("--log-dir", type=Path, default=REPOSITORY_ROOT / "data/collection_logs_vectorized")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "data/r7_vectorized_collection_manifest.json")
    parser.add_argument("--split", required=True, help="Human-readable label; not a training/validation claim.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--batch-count", type=int, required=True, help="Each batch starts one simulator with num-envs trajectories.")
    parser.add_argument("--num-envs", type=int, default=4, help="Verified default for this workstation.")
    parser.add_argument("--terrains", nargs="+", choices=DEFAULT_TERRAINS, default=DEFAULT_TERRAINS)
    parser.add_argument("--trajectory-steps", type=int, default=PAPER_ROLLOUT_STEPS)
    parser.add_argument(
        "--min-trajectory-frames",
        type=int,
        default=2,
        help="Minimum captured frames for a trajectory to make its batch training-eligible (default: 2).",
    )
    parser.add_argument("--points-per-camera", type=int, default=1500)
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for each individual IsaacLab attempt, including a resample.",
    )
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/ark/projects/IsaacLab"))
    parser.add_argument("--isaaclab-env", default="isaaclab")
    parser.add_argument(
        "--collector",
        type=Path,
        default=Path(__file__).with_name("collect_isaaclab_anymal_vectorized.py"),
        help="Vectorized IsaacLab collector (defaults to the sibling script).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--resample-failures",
        action="store_true",
        help="Schedule a fresh non-overlapping seed range when a batch misses the frame gate.",
    )
    parser.add_argument(
        "--max-resample-batches",
        type=int,
        default=0,
        help="Maximum replacement attempts per failed batch (requires --resample-failures).",
    )
    parser.add_argument(
        "--random-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass randomized velocity/yaw commands through to the vectorized collector (paper-aligned default).",
    )
    parser.add_argument("--yaw-rate-min", type=float, default=-0.35)
    parser.add_argument("--yaw-rate-max", type=float, default=0.35)
    parser.add_argument("--forward-velocity-min", type=float, default=0.5)
    parser.add_argument("--forward-velocity-max", type=float, default=1.0)
    parser.add_argument("--lateral-velocity-min", type=float, default=-0.2)
    parser.add_argument("--lateral-velocity-max", type=float, default=0.2)
    args = parser.parse_args()

    if args.trajectory_steps <= 1 or args.points_per_camera <= 0:
        raise ValueError("trajectory-steps must exceed one and points-per-camera must be positive")
    if args.min_trajectory_frames < 2 or args.min_trajectory_frames > args.trajectory_steps:
        raise ValueError("min-trajectory-frames must be at least 2 and no greater than trajectory-steps")
    if args.max_resample_batches < 0:
        raise ValueError("max-resample-batches must be non-negative")
    if args.attempt_timeout_seconds <= 0:
        raise ValueError("attempt-timeout-seconds must be positive")
    if args.yaw_rate_min >= args.yaw_rate_max or args.forward_velocity_min >= args.forward_velocity_max or args.lateral_velocity_min >= args.lateral_velocity_max:
        raise ValueError("motion lower bounds must be smaller than upper bounds")
    if args.max_resample_batches and not args.resample_failures:
        raise ValueError("max-resample-batches requires --resample-failures")
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
        "min_trajectory_frames": args.min_trajectory_frames,
        "points_per_camera": args.points_per_camera,
        "capture_schema_version": REPRODUCTION_CAPTURE_SCHEMA_VERSION,
        "terrain_profile": REPRODUCTION_TERRAIN_PROFILE,
        "random_motion": bool(args.random_motion),
        "resample_failures": bool(args.resample_failures),
        "max_resample_batches": args.max_resample_batches,
        "jobs": [],
    }

    # Queue entries retain the failed parent and attempt number.  Replacement
    # seeds are allocated after the planned seed ranges, so a retry can never
    # overwrite a prior trajectory or silently change an existing manifest.
    queue: list[tuple[VectorizedCaptureJob, str | None, int]] = [(job, None, 0) for job in jobs]
    next_resample_seed = args.seed_start + args.batch_count * args.num_envs
    processed = 0
    while queue:
        job, resample_of, resample_attempt = queue.pop(0)
        processed += 1
        record = {"job": asdict(job), "outputs": [str(path) for path in job_outputs(job, args.num_envs)], "status": "planned"}
        if resample_of is not None:
            record.update({"resample_of": resample_of, "resample_attempt": resample_attempt})
        record["job"]["output_dir"] = str(job.output_dir)
        record["job"]["log"] = str(job.log)
        if completed_job(job, args.num_envs, args.min_trajectory_frames) and not args.overwrite:
            captures = [capture_summary(path) for path in job_outputs(job, args.num_envs)]
            record.update(
                {
                    "status": "skipped_existing",
                    "finished_at": now(),
                    "captures": captures,
                    "training_eligible": [True for _ in captures],
                }
            )
            print(f"[SKIP] {job.terrain} seed={job.base_seed}", flush=True)
        else:
            command = command_for(
                args.isaaclab_root,
                args.collector,
                job,
                args.num_envs,
                args.trajectory_steps,
                args.points_per_camera,
                args.isaaclab_env,
                args.random_motion,
                args.yaw_rate_min,
                args.yaw_rate_max,
                args.forward_velocity_min,
                args.forward_velocity_max,
                args.lateral_velocity_min,
                args.lateral_velocity_max,
            )
            record["command"] = command
            if args.dry_run:
                record.update({"status": "dry_run", "finished_at": now()})
                print("[DRY] " + " ".join(command), flush=True)
            else:
                record["started_at"] = now()
                started = time.monotonic()
                environment = dict(os.environ, TERM="xterm")
                timed_out = False
                with job.log.open("w") as log_file:
                    process = subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    try:
                        returncode = process.wait(timeout=args.attempt_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        print(
                            f"[TIMEOUT] {job.terrain} seed={job.base_seed}; terminating simulator attempt",
                            flush=True,
                        )
                        terminate_process_group(process.pid)
                        try:
                            returncode = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            terminate_process_group(process.pid, force=True)
                            returncode = process.wait()
                captures = [capture_summary(path) for path in job_outputs(job, args.num_envs)]
                training_eligible = [
                    summary is not None and summary["captured_frames"] >= args.min_trajectory_frames
                    for summary in captures
                ]
                failure_kind = "timeout" if timed_out else classify_log_failure(job.log)
                record.update(
                    {
                        "status": "completed" if returncode == 0 and all(training_eligible) else "failed",
                        "returncode": returncode,
                        "duration_seconds": round(time.monotonic() - started, 1),
                        "finished_at": now(),
                        "captures": captures,
                        "training_eligible": training_eligible,
                        "failure_kind": failure_kind,
                        "timed_out": timed_out,
                    }
                )
                print(f"[{record['status'].upper()}] {job.terrain} seed={job.base_seed}", flush=True)
        manifest["jobs"].append(record)
        write_manifest(args.manifest, manifest)
        if record["status"] == "failed":
            if args.resample_failures and resample_attempt < args.max_resample_batches and record.get("failure_kind") is None:
                replacement_seed = next_resample_seed
                next_resample_seed += args.num_envs
                replacement_stem = (
                    f"vectorized_{job.terrain}_s{replacement_seed}_n{args.num_envs}"
                    f"_resample{resample_attempt + 1}"
                )
                replacement = VectorizedCaptureJob(
                    job.terrain,
                    replacement_seed,
                    args.data_dir,
                    args.log_dir / f"{replacement_stem}.log",
                )
                queue.append((replacement, f"{job.terrain}:seed={job.base_seed}", resample_attempt + 1))
                record["resample_scheduled"] = {
                    "terrain": replacement.terrain,
                    "base_seed": replacement.base_seed,
                    "attempt": resample_attempt + 1,
                }
                # Persist the scheduling decision immediately; a stopped
                # terminal should still leave a resumable manifest.
                write_manifest(args.manifest, manifest)
                print(
                    f"[RESAMPLE] {job.terrain} seed={job.base_seed} -> seed={replacement_seed} "
                    f"attempt={resample_attempt + 1}/{args.max_resample_batches}",
                    flush=True,
                )
            elif record.get("failure_kind") is not None:
                print(
                    f"[NO-RESAMPLE] {job.terrain} seed={job.base_seed}: "
                    f"simulator failure={record['failure_kind']}; clean the process/GPU before retrying",
                    file=sys.stderr,
                    flush=True,
                )
            elif not args.keep_going:
                print(f"[STOP] see {job.log}", file=sys.stderr, flush=True)
                return 1

    print(f"[DONE] {processed} vectorized batches recorded in {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
