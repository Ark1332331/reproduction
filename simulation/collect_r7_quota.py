"""Run recoverable vectorized collection until per-terrain frame quotas are met.

This thin orchestrator deliberately delegates each simulator launch to
``collect_r7_vectorized_dataset.py``.  It counts only trajectories passing the
minimum-frame gate, persists the next seed for every terrain, and can therefore
be interrupted and resumed without replacing prior NPZ files.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulation.collect_r7_dataset import capture_summary
from configs.paper_config import PAPER_ROLLOUT_STEPS, PAPER_TERRAINS

DEFAULT_TERRAINS = PAPER_TERRAINS


def now() -> str:
    return datetime.now(UTC).isoformat()


def quota_status(data_dir: Path, terrain: str, min_frames: int) -> dict[str, int]:
    """Count valid, training-eligible trajectories and frames for one terrain."""
    trajectories = 0
    eligible_frames = 0
    total_frames = 0
    for path in sorted(data_dir.glob(f"isaac_anymal_{terrain}_*.npz")):
        summary = capture_summary(path)
        if summary is None:
            continue
        frames = int(summary["captured_frames"])
        total_frames += frames
        if frames >= min_frames:
            trajectories += 1
            eligible_frames += frames
    return {
        "eligible_trajectories": trajectories,
        "eligible_frames": eligible_frames,
        "total_valid_frames": total_frames,
    }


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _collector_pids(data_dir: Path) -> list[int]:
    """Find collectors writing this quota directory, including escaped children."""
    target = str(data_dir.resolve())
    found: list[int] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
            args = [item.decode(errors="replace") for item in args if item]
        except (OSError, ValueError):
            continue
        if not any(item.endswith("collect_isaaclab_anymal_vectorized.py") for item in args):
            continue
        try:
            output_index = args.index("--output-dir")
        except ValueError:
            continue
        if output_index + 1 < len(args) and str(Path(args[output_index + 1]).resolve()) == target:
            found.append(int(proc.name))
    return found


def cleanup_stale_collectors(data_dir: Path) -> list[int]:
    """Terminate exact-output-dir collectors left behind by a failed simulator."""
    cleaned: list[int] = []
    for pid in _collector_pids(data_dir):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            cleaned.append(pid)
        except ProcessLookupError:
            continue
    if cleaned:
        # Isaac Sim may not service SIGTERM after a Vulkan/CUDA fault; give it
        # a short grace period, then force only the exact matching PIDs.
        import time

        time.sleep(2)
        for pid in cleaned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return cleaned


def cleanup_process_group(pgid: int, force: bool = False) -> None:
    """Signal a batch process group after its launcher exits or times out."""
    try:
        os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def manifest_failure_kind(path: Path) -> str | None:
    """Return the first simulator failure classification recorded by launcher."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for job in reversed(payload.get("jobs", [])):
        kind = job.get("failure_kind")
        if kind:
            return str(kind)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Recoverable per-terrain R7 frame-quota orchestrator.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--target-eligible-frames", type=int, default=4000)
    parser.add_argument("--min-trajectory-frames", type=int, default=10)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--fallback-num-envs",
        type=int,
        default=4,
        help="Use this smaller batch after a simulator GPU failure; 0 disables fallback.",
    )
    parser.add_argument("--trajectory-steps", type=int, default=PAPER_ROLLOUT_STEPS)
    parser.add_argument("--points-per-camera", type=int, default=1500)
    parser.add_argument("--max-resample-batches", type=int, default=4)
    parser.add_argument("--batch-timeout-seconds", type=int, default=300, help="Kill a stuck simulator batch after this many seconds.")
    parser.add_argument("--max-batches-per-terrain", type=int, default=0, help="0 means no per-call limit.")
    parser.add_argument("--terrains", nargs="+", choices=DEFAULT_TERRAINS, default=DEFAULT_TERRAINS)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data/vectorized_train_quota")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).parent / "data/collection_logs_vectorized_quota")
    parser.add_argument("--state", type=Path, default=Path(__file__).parent / "data/r7_quota_state.json")
    parser.add_argument("--launcher", type=Path, default=Path(__file__).with_name("collect_r7_vectorized_dataset.py"))
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/ark/projects/IsaacLab"))
    parser.add_argument("--isaaclab-env", default="isaaclab")
    parser.add_argument("--random-motion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_eligible_frames <= 0 or args.min_trajectory_frames < 2:
        raise ValueError("target-eligible-frames must be positive and min-trajectory-frames at least 2")
    if args.num_envs <= 0 or args.trajectory_steps < args.min_trajectory_frames:
        raise ValueError("num-envs must be positive and trajectory-steps must cover the frame gate")
    if args.fallback_num_envs < 0 or (args.fallback_num_envs and args.fallback_num_envs >= args.num_envs):
        raise ValueError("fallback-num-envs must be 0 or smaller than num-envs")
    if args.max_resample_batches < 0 or args.max_batches_per_terrain < 0 or args.batch_timeout_seconds <= 0:
        raise ValueError("batch limits must be non-negative")
    if not args.launcher.is_file():
        raise FileNotFoundError(f"launcher not found: {args.launcher}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    terrains = tuple(dict.fromkeys(args.terrains))
    state = {
        "created_at": now(),
        "split": args.split,
        "target_eligible_frames": args.target_eligible_frames,
        "min_trajectory_frames": args.min_trajectory_frames,
        "num_envs": args.num_envs,
        "trajectory_steps": args.trajectory_steps,
        "random_motion": bool(args.random_motion),
        "terrains": list(terrains),
        "next_seed_by_terrain": {terrain: args.seed_start for terrain in terrains},
        "batches_run": {terrain: 0 for terrain in terrains},
        "history": [],
    }
    if args.state.is_file():
        previous = json.loads(args.state.read_text())
        if previous.get("split") == args.split and tuple(previous.get("terrains", ())) == terrains:
            state = previous
    state.setdefault("num_envs_by_terrain", {terrain: args.num_envs for terrain in terrains})
    for terrain in terrains:
        state["num_envs_by_terrain"].setdefault(terrain, args.num_envs)

    # A pass visits every unfinished terrain once.  This keeps a partial,
    # interrupted collection balanced across terrain classes instead of fully
    # draining easy terrain before attempting the difficult ones.
    batches_this_call = {terrain: 0 for terrain in terrains}
    while True:
        launched = False
        for terrain in terrains:
            status = quota_status(args.data_dir, terrain, args.min_trajectory_frames)
            if status["eligible_frames"] >= args.target_eligible_frames:
                continue
            if args.max_batches_per_terrain and batches_this_call[terrain] >= args.max_batches_per_terrain:
                continue
            print(f"[QUOTA] {terrain}: {status['eligible_frames']}/{args.target_eligible_frames} eligible frames", flush=True)
            base_seed = int(state["next_seed_by_terrain"].get(terrain, args.seed_start))
            batch_num_envs = int(state["num_envs_by_terrain"].get(terrain, args.num_envs))
            manifest = args.log_dir / f"manifest_{terrain}_s{base_seed}.json"
            command = [
                sys.executable,
                str(args.launcher.resolve()),
                "--split",
                args.split,
                "--seed-start",
                str(base_seed),
                "--batch-count",
                "1",
                "--num-envs",
                str(batch_num_envs),
                "--terrains",
                terrain,
                "--trajectory-steps",
                str(args.trajectory_steps),
                "--min-trajectory-frames",
                str(args.min_trajectory_frames),
                "--points-per-camera",
                str(args.points_per_camera),
                "--attempt-timeout-seconds",
                str(args.batch_timeout_seconds),
                "--max-resample-batches",
                str(args.max_resample_batches),
                "--resample-failures",
                "--keep-going",
                "--data-dir",
                str(args.data_dir.resolve()),
                "--log-dir",
                str(args.log_dir.resolve()),
                "--manifest",
                str(manifest.resolve()),
                "--isaaclab-root",
                str(args.isaaclab_root),
                "--isaaclab-env",
                args.isaaclab_env,
            ]
            if args.random_motion:
                command.append("--random-motion")
            state["next_seed_by_terrain"][terrain] = base_seed + batch_num_envs * (1 + args.max_resample_batches)
            if args.dry_run:
                print("[DRY] " + " ".join(command), flush=True)
                continue
            stale = cleanup_stale_collectors(args.data_dir)
            if stale:
                print(f"[CLEANUP] terminated escaped collectors pids={stale}", flush=True)
            # Isaac Sim can deadlock after a Vulkan device-lost event.  Start
            # each launch in its own process group so a timeout cannot leave a
            # simulator descendant holding the GPU indefinitely.
            process = subprocess.Popen(command, start_new_session=True)
            timed_out = False
            # The launcher may perform the original attempt plus up to N
            # trajectory retries.  Each simulator attempt has its own timeout;
            # give the launcher enough wall time to finish that bounded queue.
            launcher_timeout = args.batch_timeout_seconds * (1 + args.max_resample_batches) + 30
            try:
                returncode = process.wait(timeout=launcher_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(
                    f"[TIMEOUT] {terrain} seed={base_seed}; terminating launcher after {launcher_timeout}s",
                    flush=True,
                )
                cleanup_process_group(process.pid)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    cleanup_process_group(process.pid, force=True)
                    process.wait()
                returncode = 124
            finally:
                # A launcher can return while an Isaac Sim descendant has
                # detached.  Clean the known process group, then scan by the
                # exact output directory before the next terrain starts.
                cleanup_process_group(process.pid, force=timed_out)
                stale = cleanup_stale_collectors(args.data_dir)
                if stale:
                    print(f"[CLEANUP] terminated escaped collectors pids={stale}", flush=True)
            failure_kind = manifest_failure_kind(manifest)
            if failure_kind in {"cuda_oom", "device_lost", "segmentation_fault", "timeout"}:
                if args.fallback_num_envs:
                    previous_envs = int(state["num_envs_by_terrain"].get(terrain, args.num_envs))
                    if previous_envs > args.fallback_num_envs:
                        state["num_envs_by_terrain"][terrain] = args.fallback_num_envs
                        print(
                            f"[DEGRADE] {terrain}: simulator failure={failure_kind}; "
                            f"future batches use num_envs={args.fallback_num_envs}",
                            flush=True,
                        )
            batches_this_call[terrain] += 1
            launched = True
            status = quota_status(args.data_dir, terrain, args.min_trajectory_frames)
            state["batches_run"][terrain] = int(state["batches_run"].get(terrain, 0)) + 1
            # Keep the state file useful to external monitors while a long
            # collection is still running; the final full snapshot is written
            # again after all quotas are satisfied.
            state.setdefault("final_status", {})[terrain] = status
            state["history"].append(
                {
                    "finished_at": now(),
                    "terrain": terrain,
                    "base_seed": base_seed,
                    "num_envs": batch_num_envs,
                    "returncode": returncode,
                    "failure_kind": failure_kind,
                    "timed_out": timed_out,
                    "manifest": str(manifest),
                    "status": status,
                }
            )
            write_state(args.state, state)
            print(f"[PROGRESS] {terrain}: {status['eligible_frames']}/{args.target_eligible_frames} eligible frames", flush=True)
            if returncode != 0:
                # A failed/timeout batch is itself a failed seed attempt.  The
                # next round will consume the already-reserved seed range;
                # stopping the whole quota run would defeat automatic recovery.
                print(
                    f"[RETRY] launcher failed for {terrain} seed={base_seed}; "
                    f"state saved at {args.state}",
                    file=sys.stderr,
                    flush=True,
                )
        if args.dry_run or not launched:
            break

    final = {terrain: quota_status(args.data_dir, terrain, args.min_trajectory_frames) for terrain in terrains}
    state["final_status"] = final
    if not args.dry_run:
        write_state(args.state, state)
    complete = all(item["eligible_frames"] >= args.target_eligible_frames for item in final.values())
    print(f"[DONE] quota_complete={complete} state={args.state}", flush=True)
    # A dry run intentionally cannot satisfy a quota; treat it as a successful
    # plan preview rather than a failed collection.
    return 0 if complete or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
