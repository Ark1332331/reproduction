"""Verify the paper-style per-layer likelihood/pruning contract before training.

This is deliberately not a learning experiment.  With a freshly initialized
model and paper alpha=0.5, it checks that:
  1. every intermediate likelihood BCE sees reachable positive targets; and
  2. the supervised target guard preserves every target already supported by
     the generated candidate coordinates through decoder pruning.

Targets outside a layer's unpruned candidate support are reported separately:
the guard cannot create those coordinates, so they are a generative-support
limitation rather than a pruning error.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from r5_sparse_loss import completion_loss, downsample_occupancy_target
from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import load_isaaclab_temporal_trajectory, make_autoregressive_voxels


def _load_trajectories(directory: Path, seeds: tuple[int, ...], terrains: tuple[str, ...]):
    trajectories = []
    for path in sorted(directory.glob("isaac_anymal_*.npz")):
        name = path.stem
        if any(name.endswith(f"_s{seed}") for seed in seeds) and any(terrain in name for terrain in terrains):
            trajectories.append((name, load_isaaclab_temporal_trajectory(str(path))))
    if not trajectories:
        raise RuntimeError("no trajectories matched --seeds/--terrains")
    return trajectories


def _support(candidate_coordinates: torch.Tensor, target_coordinates: np.ndarray) -> dict:
    candidates = candidate_coordinates.detach().cpu().numpy()
    k0_candidates = {tuple(row) for row in candidates[candidates[:, 4] == 0]}
    targets = {tuple(row) for row in np.asarray(target_coordinates)}
    hit_count = len(k0_candidates & targets)
    return {
        "k0_candidate_count": len(k0_candidates),
        "target_count": len(targets),
        "target_covered": hit_count,
        "target_coverage": hit_count / max(len(targets), 1),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--terrains", nargs="+", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    trajectories = _load_trajectories(args.data_dir, tuple(args.seeds), tuple(args.terrains))
    model = FourLevel4DCompletionModel().to(args.device).eval()
    # These pre-hooks observe the tensor that actually survived the preceding
    # decoder pruning and is about to feed the next upsampling layer.
    survivors = {}
    hooks = [
        model.up2.register_forward_pre_hook(lambda _module, inputs: survivors.__setitem__("l3", inputs[0].C.detach().clone())),
        model.up1.register_forward_pre_hook(lambda _module, inputs: survivors.__setitem__("l2", inputs[0].C.detach().clone())),
        model.up0.register_forward_pre_hook(lambda _module, inputs: survivors.__setitem__("l1", inputs[0].C.detach().clone())),
    ]
    factors = (("l3", 8), ("l2", 4), ("l1", 2))
    rows = []
    empty = np.empty((0, 3), dtype=float)

    try:
        with torch.no_grad():
            for trajectory_name, trajectory in trajectories:
                for frame_index, frame in enumerate(trajectory):
                    representation = make_autoregressive_voxels(
                        frame.current_measurement, empty,
                        frame.previous_to_current_translation, frame.previous_to_current_yaw,
                        voxel_size=args.voxel_size, grid_size=args.grid_size,
                    )
                    target = points_to_voxel_representation(
                        frame.target_points, empty, voxel_size=args.voxel_size, grid_size=args.grid_size,
                    )
                    target_batch = batch_voxel_representations([target])
                    target_scales = tuple(
                        torch.from_numpy(downsample_occupancy_target(target_batch, factor).coordinates)
                        for factor in (8, 4, 2, 1)
                    )
                    sparse_input = make_sparse_tensor(
                        batch_voxel_representations([representation]), device=args.device,
                    )
                    survivors.clear()
                    prediction = model(
                        sparse_input, alpha=args.alpha, training_target_coordinates=target_scales,
                    )
                    layer_rows = {}
                    for index, (name, factor) in enumerate(factors):
                        target_scale = downsample_occupancy_target(target_batch, factor)
                        likelihood = prediction.decoder_likelihoods[index]
                        dummy_offsets = torch.zeros(
                            (len(likelihood.C), 3), dtype=likelihood.F.dtype, device=likelihood.F.device,
                        )
                        likelihood_loss = completion_loss(
                            likelihood.C, likelihood.F, dummy_offsets, target_scale, position_weight=0.0,
                        )
                        before = _support(likelihood.C, target_scale.coordinates)
                        after = _support(survivors[name], target_scale.coordinates)
                        layer_rows[name] = {
                            "tensor_stride": list(likelihood.tensor_stride),
                            "unpruned": before,
                            "after_target_guarded_pruning": after,
                            "positive_count_for_bce": likelihood_loss.positive_count,
                            "bce": float(likelihood_loss.occupancy_loss),
                        }
                    final_before = _support(prediction.candidates.C, target_batch.coordinates)
                    layer_rows["l0"] = {
                        "tensor_stride": list(prediction.candidates.tensor_stride),
                        "after_final_target_guarded_pruning": final_before,
                    }
                    rows.append({"trajectory": trajectory_name, "frame": frame_index, "layers": layer_rows})
    finally:
        for hook in hooks:
            hook.remove()

    summary = {}
    for name, _ in factors:
        layer = [row["layers"][name] for row in rows]
        summary[name] = {
            "unpruned_coverage_mean": _mean([entry["unpruned"] for entry in layer], "target_coverage"),
            "surviving_coverage_mean": _mean([entry["after_target_guarded_pruning"] for entry in layer], "target_coverage"),
            "positive_count_mean": _mean(layer, "positive_count_for_bce"),
            "bce_mean": _mean(layer, "bce"),
        }
    final = [row["layers"]["l0"]["after_final_target_guarded_pruning"] for row in rows]
    summary["l0"] = {"surviving_coverage_mean": _mean(final, "target_coverage")}
    report = {
        "purpose": "paper alpha pruning contract, not a training result",
        "alpha": args.alpha,
        "model_seed": args.seed,
        "frame_count": len(rows),
        "summary": summary,
        "per_frame": rows,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, values in summary.items():
        print(f"[PROBE] {name}: {values}", flush=True)
    print(f"[DONE] wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
