"""R7 experiment on the paper-distribution ANYmal trajectories.

Loads every trajectory in ``--train-dir``, trains the four-level 4D completion
model with detached autoregressive feedback, then evaluates it on every
trajectory in ``--validation-dir`` (voxel P/R/F1 + height MAE) against the
current-measurement merge baseline. Results are written as JSON.

Usage (GPU, nsr-me-cu130-t291 env):

    python run_r7_paper_experiment.py \
        --train-dir ../reproduction/data --validation-dir ../reproduction/data \
        --split-train isaac_anymal --split-validation isaac_anymal \
        --steps 200 --results results.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    evaluate_detached_rollout,
    load_isaaclab_temporal_trajectory,
    train_detached_rollout,
)


def _load_dir(directory: Path, prefix: str, seeds: tuple[int, ...] | None, terrains: tuple[str, ...] | None):
    trajectories = []
    names = []
    for path in sorted(directory.glob(f"{prefix}_*.npz")):
        name = path.stem
        if seeds is not None and not any(f"_s{s}_" in f"{name}_" or name.endswith(f"_s{s}") for s in seeds):
            continue
        if terrains is not None and not any(t in name for t in terrains):
            continue
        trajectories.append(load_isaaclab_temporal_trajectory(str(path)))
        names.append(name)
    return trajectories, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--terrains", nargs="*", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--pruning-alpha", type=float, default=.5)
    parser.add_argument("--eval-alpha", type=float, default=None, help="Alpha for evaluation; defaults to pruning-alpha")
    parser.add_argument("--feedback-alpha", type=float, default=0.0)
    parser.add_argument("--channels", type=int, nargs=4, default=(4, 8, 12, 16))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_trajectories, train_names = _load_dir(args.train_dir, "isaac_anymal", tuple(args.train_seeds or ()), args.terrains)
    if not train_trajectories:
        raise RuntimeError(f"no training trajectories in {args.train_dir}")
    print(f"[INFO] train: {len(train_trajectories)} trajectories {train_names}")

    model = FourLevel4DCompletionModel(channels=tuple(args.channels)).to(args.device)
    run = train_detached_rollout(
        model,
        train_trajectories,
        steps=args.steps,
        voxel_size=args.voxel_size,
        grid_size=args.grid_size,
        pruning_alpha=args.pruning_alpha,
        feedback_alpha=args.feedback_alpha,
    )
    print(f"[INFO] trained {args.steps} steps; loss history: {[round(v, 4) for v in run.loss_history]}")
    print(f"[INFO] lr history: {[round(v, 6) for v in run.learning_rate_history]}")

    validation_trajectories, validation_names = _load_dir(
        args.validation_dir, "isaac_anymal", tuple(args.validation_seeds or ()), args.terrains
    )
    if not validation_trajectories:
        raise RuntimeError(f"no validation trajectories in {args.validation_dir}")

    rows = []
    for name, trajectory in zip(validation_names, validation_trajectories):
        evaluation = evaluate_detached_rollout(
            model,
            trajectory,
            voxel_size=args.voxel_size,
            grid_size=args.grid_size,
            alpha=args.eval_alpha if args.eval_alpha is not None else args.pruning_alpha,
        )
        model_occ = evaluation["autoregressive_model"]["occupancy"]
        base_occ = evaluation["current_measurement_baseline"]["occupancy"]
        model_mae = evaluation["autoregressive_model"]["height"]["mean_absolute_error"]
        base_mae = evaluation["current_measurement_baseline"]["height"]["mean_absolute_error"]
        rows.append({
            "trajectory": name,
            "model_precision": model_occ["precision"],
            "model_recall": model_occ["recall"],
            "model_f1": model_occ["f1"],
            "baseline_precision": base_occ["precision"],
            "baseline_recall": base_occ["recall"],
            "baseline_f1": base_occ["f1"],
            "model_height_mae": model_mae,
            "baseline_height_mae": base_mae,
        })
        print(f"[eval] {name}: model F1={model_occ['f1']:.4f} (baseline {base_occ['f1']:.4f}) "
              f"MAE={model_mae:.4f} (baseline {base_mae:.4f})")

    def _mean(key):
        return statistics.mean(row[key] for row in rows)

    report = {
        "config": {
            "train_trajectories": train_names,
            "validation_trajectories": validation_names,
            "steps": args.steps,
            "voxel_size": args.voxel_size,
            "grid_size": args.grid_size,
            "pruning_alpha": args.pruning_alpha,
            "channels": list(args.channels),
            "seed": args.seed,
        },
        "train_loss_first": run.loss_history[0],
        "train_loss_last": run.loss_history[-1],
        "per_trajectory": rows,
        "mean": {
            "model_f1": _mean("model_f1"),
            "baseline_f1": _mean("baseline_f1"),
            "model_height_mae": _mean("model_height_mae"),
            "baseline_height_mae": _mean("baseline_height_mae"),
            "model_precision": _mean("model_precision"),
            "model_recall": _mean("model_recall"),
            "baseline_precision": _mean("baseline_precision"),
            "baseline_recall": _mean("baseline_recall"),
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[INFO] wrote {args.results}")
    if args.checkpoint:
        torch.save(model.state_dict(), args.checkpoint)
        print(f"[INFO] saved checkpoint {args.checkpoint}")


if __name__ == "__main__":
    main()
