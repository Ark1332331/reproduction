"""Evaluate a trained checkpoint on the R7 paper-distribution validation set.

Loads a checkpoint produced by run_r7_paper_train.py and evaluates the model on
every validation trajectory (voxel P/R/F1 + height MAE) against the
current-measurement merge baseline, optionally scanning evaluation alphas.

Usage (GPU, nsr-me-cu130-t291 env):

    python run_r7_paper_eval.py --validation-dir ../reproduction/data \
        --checkpoint r7_train_a0.pt --alphas 0.05 0.1 0.2 0.3 0.5 \
        --results eval.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import evaluate_detached_rollout, load_isaaclab_temporal_trajectory
from paper_config import PAPER_GRID_SIZE, PAPER_PRUNING_ALPHA, PAPER_TERRAINS, PAPER_VOXEL_SIZE_M


def _load_dir(
    directory: Path,
    prefix: str,
    seeds: tuple[int, ...] | None,
    terrains: tuple[str, ...] | None,
    min_trajectory_frames: int = 10,
    max_trajectories_per_terrain: int | None = None,
):
    trajectories = []
    names = []
    terrain_counts = {terrain: 0 for terrain in (terrains or ())}
    for path in sorted(directory.glob(f"{prefix}_*.npz")):
        name = path.stem
        if seeds is not None and not any(f"_s{s}_" in f"{name}_" or name.endswith(f"_s{s}") for s in seeds):
            continue
        if terrains is not None and not any(t in name for t in terrains):
            continue
        terrain = next((t for t in (terrains or ()) if f"_{t}_" in name), None)
        if terrain is not None and max_trajectories_per_terrain is not None:
            if terrain_counts[terrain] >= max_trajectories_per_terrain:
                continue
        trajectory = load_isaaclab_temporal_trajectory(str(path))
        if len(trajectory) < min_trajectory_frames:
            continue
        trajectories.append(trajectory)
        names.append(name)
        if terrain is not None:
            terrain_counts[terrain] += 1
    return trajectories, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--terrains", nargs="*", default=PAPER_TERRAINS)
    parser.add_argument("--min-trajectory-frames", type=int, default=10)
    parser.add_argument(
        "--max-trajectories-per-terrain", type=int, default=None,
        help="Optional deterministic per-terrain cap for fast smoke evaluation.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="*", default=(PAPER_PRUNING_ALPHA,))
    parser.add_argument(
        "--disable-internal-pruning",
        action="store_true",
        help="diagnostic: keep decoder candidates, but still threshold final predictions by --alphas",
    )
    parser.add_argument(
        "--feedback-alpha", type=float, default=None,
        help="alpha selecting the next frame's history (previous_estimate); "
        "defaults to the sweep alpha for backward compatibility",
    )
    parser.add_argument(
        "--likelihood-logit-offset", type=float, default=0.0,
        help="Optional post-hoc logit calibration added before alpha thresholding; default is raw logits",
    )
    parser.add_argument(
        "--disable-history",
        action="store_true",
        help="ablation: evaluate every frame with an empty k=1 history",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=PAPER_VOXEL_SIZE_M)
    parser.add_argument("--grid-size", type=int, default=PAPER_GRID_SIZE)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = state.get("config", {})
    kernels = config.get("generative_kernels")
    if kernels is not None:
        kernels = tuple(tuple(int(v) for v in kernel) for kernel in kernels)
    model_kwargs = {"channels": tuple(config.get("channels", (4, 8, 12, 16)))}
    # ``None`` means the normal k2 architecture. Do not pass it explicitly:
    # the model validates a concrete four-kernel tuple when this argument exists.
    if kernels is not None:
        model_kwargs["generative_kernels"] = kernels
    model = FourLevel4DCompletionModel(**model_kwargs).to(args.device)
    model.load_state_dict(state["model"])
    print(f"[INFO] loaded checkpoint {args.checkpoint} (trained to step {state.get('step')}, "
          f"channels {config.get('channels')}, generative_kernels {kernels}, loss history tail "
          f"{[round(v, 3) for v in state.get('loss_history', [])[-5:]]})", flush=True)

    trajectories, names = _load_dir(
        args.validation_dir, "isaac_anymal", None if args.seeds is None else tuple(args.seeds), args.terrains,
        args.min_trajectory_frames, args.max_trajectories_per_terrain,
    )
    if not trajectories:
        raise RuntimeError(f"no validation trajectories in {args.validation_dir}")
    print(f"[INFO] validation: {len(trajectories)} trajectories {names}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "metric_protocol": {
            "paper_primary_aggregation": "macro_per_frame",
            "micro_aggregation": "global_count_aggregate",
            "height_error": "highest_z_per_xy_cell_on_shared_xy_cells",
            "height_reports_coverage": True,
            "height_unit": "meters",
        },
        "trained_steps": state.get("step"),
        "config": config,
        "internal_pruning": not args.disable_internal_pruning,
        "feedback_alpha": args.feedback_alpha,
        "likelihood_logit_offset": args.likelihood_logit_offset,
        "use_history": not args.disable_history,
        "max_trajectories_per_terrain": args.max_trajectories_per_terrain,
    }
    for alpha in args.alphas:
        rows = []
        for name, trajectory in zip(names, trajectories):
            evaluation = evaluate_detached_rollout(
                model,
                trajectory,
                voxel_size=args.voxel_size,
                grid_size=args.grid_size,
                alpha=alpha,
                prune_internal=not args.disable_internal_pruning,
                feedback_alpha=args.feedback_alpha,
                use_history=not args.disable_history,
                likelihood_logit_offset=args.likelihood_logit_offset,
            )
            model_summary = evaluation["autoregressive_model"]
            base_summary = evaluation["current_measurement_baseline"]
            model_occ = model_summary["macro"]["occupancy"]
            base_occ = base_summary["macro"]["occupancy"]
            model_micro_occ = model_summary["micro"]["occupancy"]
            base_micro_occ = base_summary["micro"]["occupancy"]
            model_height = model_summary["macro"]["height"]
            base_height = base_summary["macro"]["height"]
            rows.append({
                "trajectory": name,
                "model_f1": model_occ["f1"],
                "model_precision": model_occ["precision"],
                "model_recall": model_occ["recall"],
                "baseline_f1": base_occ["f1"],
                "baseline_precision": base_occ["precision"],
                "baseline_recall": base_occ["recall"],
                "model_micro_f1": model_micro_occ["f1"],
                "baseline_micro_f1": base_micro_occ["f1"],
                "model_height_mae": model_height["mean_absolute_error"],
                "baseline_height_mae": base_height["mean_absolute_error"],
                "model_height_coverage": model_height["coverage"],
                "baseline_height_coverage": base_height["coverage"],
                "model_height_matched_cell_count": model_height["matched_cell_count"],
                "baseline_height_matched_cell_count": base_height["matched_cell_count"],
            })
            print(f"[alpha={alpha}] {name}: macro model F1={model_occ['f1']:.4f} "
                  f"(baseline {base_occ['f1']:.4f}) MAE={model_height['mean_absolute_error']:.4f} "
                  f"coverage={model_height['coverage']:.3f}", flush=True)

        def _mean(key):
            return statistics.mean(row[key] for row in rows)

        report[f"alpha_{alpha}"] = {
            "mean": {
                "model_f1": _mean("model_f1"),
                "baseline_f1": _mean("baseline_f1"),
                "model_precision": _mean("model_precision"),
                "model_recall": _mean("model_recall"),
                "model_height_mae": _mean("model_height_mae"),
                "baseline_height_mae": _mean("baseline_height_mae"),
                "model_height_coverage": _mean("model_height_coverage"),
                "baseline_height_coverage": _mean("baseline_height_coverage"),
                "baseline_precision": _mean("baseline_precision"),
                "baseline_recall": _mean("baseline_recall"),
            },
            "per_trajectory": rows,
        }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[INFO] wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
