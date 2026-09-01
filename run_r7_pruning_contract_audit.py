"""Audit whether every decoder scale can support paper-style likelihood pruning.

For each unpruned decoder tensor, compare its k=0 candidate coordinates against
the ground-truth occupancy target at that scale.  A target voxel outside a
candidate tensor cannot receive BCE supervision or survive target-guarded
pruning, so this script separates a candidate-support failure from an alpha
threshold failure.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from r5_sparse_loss import downsample_occupancy_target
from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import load_isaaclab_temporal_trajectory, make_autoregressive_voxels


def _load_trajectories(directory: Path, seeds: tuple[int, ...], terrains: tuple[str, ...]):
    matches = []
    for path in sorted(directory.glob("isaac_anymal_*.npz")):
        name = path.stem
        if any(name.endswith(f"_s{seed}") for seed in seeds) and any(terrain in name for terrain in terrains):
            matches.append((path.stem, load_isaaclab_temporal_trajectory(str(path))))
    if not matches:
        raise RuntimeError("no trajectories matched --seeds/--terrains")
    return matches


def _layer_summary(candidate_coordinates: torch.Tensor, target_coordinates: np.ndarray) -> dict:
    candidates = candidate_coordinates.detach().cpu().numpy().astype(np.int32)
    target = np.asarray(target_coordinates, dtype=np.int32)
    k_counts = Counter(int(value) for value in candidates[:, 4])
    k0 = candidates[candidates[:, 4] == 0]
    target_set = {tuple(row) for row in target}
    k0_set = {tuple(row) for row in k0}
    hits = len(target_set & k0_set)
    return {
        "candidate_count": int(len(candidates)),
        "k0_candidate_count": int(len(k0)),
        "candidate_k_counts": dict(sorted(k_counts.items())),
        "target_count": int(len(target_set)),
        "target_covered": hits,
        "target_coverage": hits / max(len(target_set), 1),
        "positive_rate_among_k0_candidates": hits / max(len(k0), 1),
    }


def _direct_divide_target(target_coordinates: np.ndarray, spatial_factor: int) -> np.ndarray:
    """Historical, incorrect target convention retained only for audit comparison."""
    coordinates = np.asarray(target_coordinates, dtype=np.int32).copy()
    coordinates[:, 1:4] //= spatial_factor
    return np.unique(coordinates, axis=0)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--terrains", nargs="+", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    trajectories = _load_trajectories(args.data_dir, tuple(args.seeds), tuple(args.terrains))
    model = FourLevel4DCompletionModel().to(args.device).eval()
    aggregates = defaultdict(list)
    per_frame = []
    previous_empty = np.empty((0, 3), dtype=float)

    with torch.no_grad():
        for name, trajectory in trajectories:
            for frame_index, frame in enumerate(trajectory):
                # Candidate support is measured before recurrence effects: the
                # current measurement alone supplies the conservative lower bound.
                representation = make_autoregressive_voxels(
                    frame.current_measurement,
                    previous_empty,
                    frame.previous_to_current_translation,
                    frame.previous_to_current_yaw,
                    voxel_size=args.voxel_size,
                    grid_size=args.grid_size,
                )
                target = points_to_voxel_representation(
                    frame.target_points,
                    previous_empty,
                    voxel_size=args.voxel_size,
                    grid_size=args.grid_size,
                )
                target_batch = batch_voxel_representations([target])
                input_tensor = make_sparse_tensor(batch_voxel_representations([representation]), device=args.device)
                prediction = model(input_tensor, alpha=None)
                scales = (
                    ("l3_stride8", prediction.decoder_likelihoods[0], 8),
                    ("l2_stride4", prediction.decoder_likelihoods[1], 4),
                    ("l1_stride2", prediction.decoder_likelihoods[2], 2),
                    ("l0_stride1", prediction.candidates, 1),
                )
                row = {"trajectory": name, "frame": frame_index, "layers": {}}
                for layer_name, sparse_layer, factor in scales:
                    layer_target = downsample_occupancy_target(target_batch, factor).coordinates
                    values = _layer_summary(sparse_layer.C, layer_target)
                    historical = _layer_summary(
                        sparse_layer.C,
                        _direct_divide_target(target_batch.coordinates, factor),
                    )
                    values["tensor_stride"] = list(sparse_layer.tensor_stride)
                    values["historical_direct_divide_coverage"] = historical["target_coverage"]
                    row["layers"][layer_name] = values
                    aggregates[layer_name].append(values)
                per_frame.append(row)

    summary = {}
    for layer_name, rows in aggregates.items():
        summary[layer_name] = {
            "target_coverage_mean": _mean([row["target_coverage"] for row in rows]),
            "target_coverage_min": min(row["target_coverage"] for row in rows),
            "historical_direct_divide_coverage_mean": _mean(
                [row["historical_direct_divide_coverage"] for row in rows]
            ),
            "positive_rate_mean": _mean([row["positive_rate_among_k0_candidates"] for row in rows]),
            "k0_candidates_mean": _mean([row["k0_candidate_count"] for row in rows]),
            "target_count_mean": _mean([row["target_count"] for row in rows]),
        }
    report = {
        "purpose": "unpruned per-layer candidate support for paper-style likelihood/pruning",
        "history_input": "empty (conservative current-measurement lower bound)",
        "trajectories": [name for name, _ in trajectories],
        "summary": summary,
        "per_frame": per_frame,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for layer, values in summary.items():
        print(f"[AUDIT] {layer}: coverage mean={values['target_coverage_mean']:.4f}, "
              f"min={values['target_coverage_min']:.4f}, positive rate={values['positive_rate_mean']:.4f}", flush=True)
    print(f"[DONE] wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
