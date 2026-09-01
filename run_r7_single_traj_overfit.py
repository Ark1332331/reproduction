"""P3-d single-trajectory sequence overfit.

Fixes one 12-step trajectory, trains with the validated setting (pos_weight=1,
likelihood off, pruning 0, feedback 0). Every 20 steps it reports the whole
sequence's P/R/F1, height MAE, per-frame predicted count and per-frame FP/FN via
the SAME evaluation path (detached feedback, feedback_alpha=0, no internal
pruning). Success = sequence F1 approaches the trajectory's coverage ceiling.

Usage:

    python run_r7_single_traj_overfit.py --data-dir ../reproduction/data --steps 400
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    evaluate_detached_rollout,
    make_autoregressive_voxels,
    batch_voxel_representations,
    make_sparse_tensor,
    points_to_voxel_representation,
)
from r5_sparse_loss import completion_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=Path("single_traj_overfit.pt"))
    args = parser.parse_args()

    trajectory = load_isaaclab_temporal_trajectory(str(args.data_dir / "isaac_anymal_boxes_s10.npz"))
    print(f"trajectory frames: {len(trajectory)}", flush=True)
    model = FourLevel4DCompletionModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for it in range(args.steps):
        model.train()
        optimizer.zero_grad()
        previous_estimate = np.empty((0, 3), dtype=float)
        losses = []
        for step in trajectory:
            target = points_to_voxel_representation(step.target_points, np.empty((0, 3), dtype=float))
            target_batch = batch_voxel_representations([target])
            input_rep = make_autoregressive_voxels(
                step.current_measurement, previous_estimate,
                step.previous_to_current_translation, step.previous_to_current_yaw,
            )
            input_tensor = make_sparse_tensor(batch_voxel_representations([input_rep]), device=args.device)
            pred = model(input_tensor, alpha=None)
            loss = completion_loss(
                pred.candidates.C, pred.occupancy_logits.F, pred.position_offsets.F, target_batch,
                occupancy_pos_weight=1,
            )
            loss.total.backward()
            losses.append(float(loss.total))
            with torch.no_grad():
                feedback = model(input_tensor, alpha=None)
                previous_estimate = _points_from(feedback, alpha=0.0)
        optimizer.step()

        if (it + 1) % 20 == 0:
            evaluation = evaluate_detached_rollout(
                model, trajectory, alpha=0.1, prune_internal=False, feedback_alpha=0.0
            )
            occ = evaluation["autoregressive_model"]["occupancy"]
            h = evaluation["autoregressive_model"]["height"]
            base = evaluation["current_measurement_baseline"]["occupancy"]
            print(
                f"[step {it+1:03d}] loss={np.mean(losses):.4f} "
                f"model F1={occ['f1']:.4f} P={occ['precision']:.4f} R={occ['recall']:.4f} MAE={h['mean_absolute_error']:.4f} "
                f"(baseline F1={base['f1']:.4f})",
                flush=True,
            )


    torch.save(
        {
            "model": model.state_dict(),
            "step": args.steps,
            "config": {
                "channels": [4, 8, 12, 16],
                "generative_kernels": None,  # k2 default
                "pruning_alpha": 0.0,
                "feedback_alpha": 0.0,
                "occupancy_pos_weight": 1,
                "likelihood_weight": 0,
                "trajectory": "isaac_anymal_boxes_s10.npz",
            },
        },
        args.checkpoint,
    )
    print(f"[INFO] saved {args.checkpoint}", flush=True)


def _points_from(pred, alpha: float) -> np.ndarray:
    from r7_autoregressive_rollout import sparse_prediction_to_current_points

    return sparse_prediction_to_current_points(
        pred.candidates.C, pred.occupancy_logits.F, pred.position_offsets.F,
        alpha=alpha, voxel_size=0.05,
    )


if __name__ == "__main__":
    main()
