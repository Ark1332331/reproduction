"""Candidate coverage audit for the paper-distribution data (external review P1-3).

Generative-transpose candidate coordinates depend only on the input coordinates
and the kernel support (not on the trained weights), so a randomly initialised
model suffices. For every validation trajectory frame we report three variants:

  coverage_k0_only : k=1 empty (pure current measurement)
  coverage_k1_previous_observation : k=1 = the previous raw camera
                                      observation, pose-aligned into the
                                      current frame. This is a diagnostic
                                      proxy, not the paper's runtime input.
  coverage_k1_oracle_target : k=1 = the previous complete target scene,
                              pose-aligned into the current frame. This is an
                              oracle upper bound; the paper uses a previous
                              model estimate, which must lie between cases.

Coverage < 1 is a hard recall ceiling: target voxels without any candidate can
never be predicted. Comparing the two variants shows whether history actually
widens the candidate support (temporal kernel value) or not (generative kernel
support problem).

Usage (GPU, nsr-me-cu130-t291 env):

    python run_coverage_audit.py --data-dir ../reproduction/data --results coverage.json
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    make_autoregressive_voxels,
    batch_voxel_representations,
    make_sparse_tensor,
    points_to_voxel_representation,
)


def _coverage(model, step, device, voxel_size, grid_size, previous_points):
    inp = make_autoregressive_voxels(
        step.current_measurement,
        previous_points,
        step.previous_to_current_translation,
        step.previous_to_current_yaw,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    target = points_to_voxel_representation(step.target_points, np.empty((0, 3), dtype=float))
    tensor = make_sparse_tensor(batch_voxel_representations([inp]), device=device)
    target_keys = {(0,) + tuple(map(int, row)) for row in target.coords}
    with torch.inference_mode():
        pred = model(tensor, alpha=None)
    cand_keys = {tuple(int(v) for v in row) for row in pred.candidates.C.tolist()}
    covered = len(cand_keys & target_keys)
    return covered / len(target_keys), len(target_keys), len(cand_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=(10, 11, 12, 13))
    parser.add_argument("--terrains", nargs="*", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--results", type=Path, default=Path("coverage_audit.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--generative-kernels", type=int, nargs=16, default=None,
        help="Four 4D generative kernels (16 ints) for up3/up2/up1/up0; default (2,2,2,2) x4",
    )
    args = parser.parse_args()

    kernels = None
    if args.generative_kernels is not None:
        kernels = tuple(tuple(args.generative_kernels[i : i + 4]) for i in range(0, 16, 4))
        print(f"[INFO] generative kernels: {kernels}", flush=True)
    model = FourLevel4DCompletionModel(generative_kernels=kernels).to(args.device) if kernels else FourLevel4DCompletionModel().to(args.device)
    model.eval()
    rows = []
    for terrain in args.terrains:
        for seed in args.seeds:
            path = args.data_dir / f"isaac_anymal_{terrain}_s{seed}.npz"
            if not path.exists():
                continue
            trajectory = load_isaaclab_temporal_trajectory(str(path))
            previous_observation = np.empty((0, 3), dtype=float)
            previous_target = np.empty((0, 3), dtype=float)
            for frame_index, step in enumerate(trajectory):
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                cov_k0, target_n, cand_n = _coverage(
                    model, step, args.device, 0.05, 64, np.empty((0, 3), dtype=float)
                )
                cov_observation, _, _ = _coverage(
                    model, step, args.device, 0.05, 64, previous_observation
                )
                cov_oracle, _, _ = _coverage(
                    model, step, args.device, 0.05, 64, previous_target
                )
                rows.append({
                    "trajectory": path.stem,
                    "frame": frame_index,
                    "coverage_k0_only": round(cov_k0, 4),
                    "coverage_k1_previous_observation": round(cov_observation, 4),
                    "coverage_k1_oracle_target": round(cov_oracle, 4),
                    "target_voxels": target_n,
                    "candidate_voxels": cand_n,
                })
                print(f"{path.stem} f{frame_index:02d}: k0-only {cov_k0:.3f} | "
                      f"previous-observation {cov_observation:.3f} | oracle-target {cov_oracle:.3f} "
                      f"({target_n} targets, {cand_n} candidates)", flush=True)
                previous_observation = step.current_measurement
                previous_target = step.target_points

    report = {
        "summary": {
            "k0_only_mean": round(statistics.mean(r["coverage_k0_only"] for r in rows), 4),
            "k0_only_min": round(min(r["coverage_k0_only"] for r in rows), 4),
            "k1_previous_observation_mean": round(
                statistics.mean(r["coverage_k1_previous_observation"] for r in rows), 4
            ),
            "k1_previous_observation_min": round(
                min(r["coverage_k1_previous_observation"] for r in rows), 4
            ),
            "k1_oracle_target_mean": round(
                statistics.mean(r["coverage_k1_oracle_target"] for r in rows), 4
            ),
            "k1_oracle_target_min": round(
                min(r["coverage_k1_oracle_target"] for r in rows), 4
            ),
        },
        "per_frame": rows,
    }
    args.results.write_text(json.dumps(report, indent=2))
    print(f"[INFO] wrote {args.results}")
    print(f"[SUMMARY] k0-only mean {report['summary']['k0_only_mean']:.4f} "
          f"min {report['summary']['k0_only_min']:.4f} | previous-observation mean "
          f"{report['summary']['k1_previous_observation_mean']:.4f} min "
          f"{report['summary']['k1_previous_observation_min']:.4f} | oracle-target mean "
          f"{report['summary']['k1_oracle_target_mean']:.4f} min "
          f"{report['summary']['k1_oracle_target_min']:.4f}")


if __name__ == "__main__":
    main()
