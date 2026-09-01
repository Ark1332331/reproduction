"""Compare trained R7 checkpoints without changing their weights.

The audit replays the same detached autoregressive inputs used at training time
(``feedback_alpha=0`` by default).  It records whether different random seeds
separate positive and negative final candidates, and how dense their k=1
feedback becomes.  It is a diagnostic, not an evaluation score selector.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from r5_sparse_loss import completion_loss
from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    make_autoregressive_voxels,
    sparse_prediction_to_current_points,
)


def _load_trajectories(directory: Path, seeds: tuple[int, ...], terrains: tuple[str, ...]):
    paths = []
    for path in sorted(directory.glob("isaac_anymal_*.npz")):
        name = path.stem
        if not any(name.endswith(f"_s{seed}") for seed in seeds):
            continue
        if not any(terrain in name for terrain in terrains):
            continue
        paths.append(path)
    if not paths:
        raise RuntimeError("no trajectories matched --seeds/--terrains")
    return [(path.stem, load_isaaclab_temporal_trajectory(str(path))) for path in paths]


def _make_model(checkpoint: Path, device: str):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    kernels = config.get("generative_kernels")
    kwargs = {"channels": tuple(config.get("channels", (4, 8, 12, 16)))}
    if kernels is not None:
        kwargs["generative_kernels"] = tuple(tuple(int(value) for value in kernel) for kernel in kernels)
    model = FourLevel4DCompletionModel(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def audit_checkpoint(
    checkpoint: Path,
    trajectories,
    device: str,
    voxel_size: float,
    grid_size: int,
    feedback_alpha: float,
) -> dict:
    model, state = _make_model(checkpoint, device)
    positive_probabilities: list[float] = []
    negative_probabilities: list[float] = []
    positive_bces: list[float] = []
    negative_bces: list[float] = []
    per_frame = []

    with torch.no_grad():
        for trajectory_name, trajectory in trajectories:
            previous_estimate = np.empty((0, 3), dtype=float)
            for frame_index, frame in enumerate(trajectory):
                representation = make_autoregressive_voxels(
                    frame.current_measurement,
                    previous_estimate,
                    frame.previous_to_current_translation,
                    frame.previous_to_current_yaw,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                )
                target_representation = points_to_voxel_representation(
                    frame.target_points,
                    np.empty((0, 3), dtype=float),
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                )
                sparse_input = make_sparse_tensor(
                    batch_voxel_representations([representation]), device=device
                )
                target_batch = batch_voxel_representations([target_representation])
                prediction = model(sparse_input, alpha=None)
                loss = completion_loss(
                    prediction.candidates.C,
                    prediction.occupancy_logits.F,
                    prediction.position_offsets.F,
                    target_batch,
                    occupancy_pos_weight=None,
                )
                logits = prediction.occupancy_logits.F[:, 0]
                labels = loss.occupancy_targets[:, 0].bool()
                probabilities = torch.sigmoid(logits)
                positive_values = probabilities[labels].cpu().tolist()
                negative_values = probabilities[~labels].cpu().tolist()
                positive_probabilities.extend(positive_values)
                negative_probabilities.extend(negative_values)
                if bool(labels.any()):
                    positive_bces.append(float(F.binary_cross_entropy_with_logits(logits[labels], torch.ones_like(logits[labels]))))
                if bool((~labels).any()):
                    negative_bces.append(float(F.binary_cross_entropy_with_logits(logits[~labels], torch.zeros_like(logits[~labels]))))
                previous_estimate = sparse_prediction_to_current_points(
                    prediction.candidates.C,
                    prediction.occupancy_logits.F,
                    prediction.position_offsets.F,
                    alpha=feedback_alpha,
                    voxel_size=voxel_size,
                )
                # R1 coordinates are [x, y, z, k]; batch is added only when
                # constructing the Minkowski sparse tensor.
                input_coordinates = representation.coords
                per_frame.append(
                    {
                        "trajectory": trajectory_name,
                        "frame": frame_index,
                        "input_k0_voxels": int((input_coordinates[:, 3] == 0).sum()),
                        "input_k1_voxels": int((input_coordinates[:, 3] == 1).sum()),
                        "target_voxels": int(len(target_batch.coordinates)),
                        "final_candidates": int(len(prediction.candidates.C)),
                        "positive_candidates": int(labels.sum().item()),
                        "feedback_points": int(len(previous_estimate)),
                        "positive_probability_mean": _mean(positive_values),
                        "negative_probability_mean": _mean(negative_values),
                    }
                )

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": state.get("step"),
        "config": state.get("config", {}),
        "feedback_alpha": feedback_alpha,
        "summary": {
            "positive_probability": _distribution(positive_probabilities),
            "negative_probability": _distribution(negative_probabilities),
            "positive_bce_mean": _mean(positive_bces),
            "negative_bce_mean": _mean(negative_bces),
            "input_k1_voxels_mean": _mean([row["input_k1_voxels"] for row in per_frame]),
            "feedback_points_mean": _mean([row["feedback_points"] for row in per_frame]),
            "final_candidates_mean": _mean([row["final_candidates"] for row in per_frame]),
            "positive_rate": _mean(
                [row["positive_candidates"] / max(row["final_candidates"], 1) for row in per_frame]
            ),
        },
        "per_frame": per_frame,
    }
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--terrains", nargs="+", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--feedback-alpha", type=float, default=0.0)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    trajectories = _load_trajectories(args.data_dir, tuple(args.seeds), tuple(args.terrains))
    report = {
        "purpose": "compare checkpoint label separation and feedback density under one fixed rollout contract",
        "trajectories": [name for name, _ in trajectories],
        "checkpoints": [
            audit_checkpoint(
                checkpoint,
                trajectories,
                args.device,
                args.voxel_size,
                args.grid_size,
                args.feedback_alpha,
            )
            for checkpoint in args.checkpoints
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for result in report["checkpoints"]:
        summary = result["summary"]
        print(
            f"[AUDIT] {Path(result['checkpoint']).name}: pos p={summary['positive_probability']['mean']:.4f}, "
            f"neg p={summary['negative_probability']['mean']:.4f}, "
            f"k1={summary['input_k1_voxels_mean']:.0f}, feedback={summary['feedback_points_mean']:.0f}",
            flush=True,
        )
    print(f"[DONE] wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
