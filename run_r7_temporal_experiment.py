"""Run a reproducible detached autoregressive rollout on one R7 trajectory."""

import argparse
import json
from pathlib import Path

import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    evaluate_detached_rollout,
    load_isaaclab_temporal_trajectory,
    train_detached_rollout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--validation-trajectory")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--pruning-alpha", type=float, default=.5)
    parser.add_argument("--feedback-alpha", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--final-learning-rate", type=float, default=.0001)
    parser.add_argument("--channels", type=int, nargs=4, default=(4, 8, 12, 16))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    trajectory = load_isaaclab_temporal_trajectory(args.trajectory)
    model = FourLevel4DCompletionModel(channels=tuple(args.channels)).to(args.device)
    run = train_detached_rollout(
        model,
        [trajectory],
        steps=args.steps,
        voxel_size=args.voxel_size,
        grid_size=args.grid_size,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        pruning_alpha=args.pruning_alpha,
        feedback_alpha=args.feedback_alpha,
    )
    train_evaluation = evaluate_detached_rollout(
        model, trajectory, voxel_size=args.voxel_size, grid_size=args.grid_size, alpha=args.pruning_alpha
    )
    validation_evaluation = None
    if args.validation_trajectory:
        validation_trajectory = load_isaaclab_temporal_trajectory(args.validation_trajectory)
        validation_evaluation = evaluate_detached_rollout(
            model, validation_trajectory, voxel_size=args.voxel_size, grid_size=args.grid_size, alpha=args.pruning_alpha
        )
    report = {
        "trajectory": args.trajectory,
        "trajectory_frames": len(trajectory),
        "config": vars(args) | {
            "results": str(args.results) if args.results else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
        "training": {
            "initial_loss": run.initial_loss,
            "final_loss": run.final_loss,
            "loss_history": list(run.loss_history),
            "learning_rate_history": list(run.learning_rate_history),
        },
        "evaluation": {"train": train_evaluation, "validation": validation_evaluation},
        "scope": "temporal rollout diagnostic; no paper-performance claim without many held-out trajectories",
    }
    print(f"trajectory_frames={len(trajectory)} train_loss={run.initial_loss:.6f}->{run.final_loss:.6f}")
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved results: {args.results}")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
