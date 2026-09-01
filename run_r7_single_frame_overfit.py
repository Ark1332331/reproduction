"""P3-b single-frame overfit with pos_weight ablation (1 / auto / 20).

Fixes one frame, no history, likelihood off. Every 20 steps records total loss,
positive/negative BCE, TP/FP/FN/P/R/F1, positive/negative sigmoid quantiles
(p10/p50/p90/mean) and the predicted-occupied count. Success = fixed-frame F1
rises and FP keeps falling (not just loss falling).

Usage:

    python run_r7_single_frame_overfit.py --pos-weight 1|auto|20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from r5_sparse_model import FourLevel4DCompletionModel
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    make_autoregressive_voxels,
    batch_voxel_representations,
    make_sparse_tensor,
    points_to_voxel_representation,
)
from r5_sparse_loss import completion_loss


def _quantiles(values, qs=(0.1, 0.5, 0.9)):
    if len(values) == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(values, q)) for q in qs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pos-weight", default="20")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    trajectory = load_isaaclab_temporal_trajectory(str(args.data_dir / "isaac_anymal_boxes_s10.npz"))
    step = trajectory[0]
    target = points_to_voxel_representation(step.target_points, np.empty((0, 3), dtype=float))
    target_batch = batch_voxel_representations([target])
    target_keys = {(0,) + tuple(map(int, r)) for r in target.coords}
    input_rep = make_autoregressive_voxels(
        step.current_measurement, np.empty((0, 3), dtype=float),
        step.previous_to_current_translation, step.previous_to_current_yaw,
    )
    input_tensor = make_sparse_tensor(batch_voxel_representations([input_rep]), device=args.device)

    model = FourLevel4DCompletionModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    print(f"pos_weight={args.pos_weight} target_voxels={len(target_keys)}", flush=True)

    for it in range(args.steps):
        model.train()
        optimizer.zero_grad()
        pred = model(input_tensor, alpha=None)
        loss = completion_loss(
            pred.candidates.C, pred.occupancy_logits.F, pred.position_offsets.F, target_batch,
            occupancy_pos_weight=args.pos_weight,
        )
        loss.total.backward()
        optimizer.step()

        if (it + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred_full = model(input_tensor, alpha=None)
            probs = torch.sigmoid(pred_full.occupancy_logits.F[:, 0]).detach().cpu().numpy()
            coords = pred_full.candidates.C.detach().cpu().numpy()
            keys = [tuple(int(v) for v in r) for r in coords]
            on = np.array([k in target_keys for k in keys])
            # positive / negative sigmoid quantiles
            pos_q = _quantiles(probs[on]) if on.any() else [float("nan")] * 3
            neg_q = _quantiles(probs[~on]) if (~on).any() else [float("nan")] * 3
            pos_mean = float(probs[on].mean()) if on.any() else 0.0
            neg_mean = float(probs[~on].mean()) if (~on).any() else 0.0
            # BCE decomposition (no pos_weight, raw)
            labels = torch.tensor(on, dtype=torch.float32, device=args.device).unsqueeze(1)
            logits = pred_full.occupancy_logits.F
            pos_bce = float(F.binary_cross_entropy_with_logits(logits[on], labels[on])) if on.any() else 0.0
            neg_bce = float(F.binary_cross_entropy_with_logits(logits[~on], labels[~on])) if (~on).any() else 0.0
            # threshold selection (k0 only, sigmoid >= 0.5)
            keep = (probs >= 0.5)
            pred_keys = {k for k, m in zip(keys, keep) if m}
            tp = len(pred_keys & target_keys)
            fp = len(pred_keys - target_keys)
            fn = len(target_keys - pred_keys)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print(
                f"[step {it+1:03d}] loss={float(loss.total):.4f} posB={pos_bce:.4f} negB={neg_bce:.4f} "
                f"F1={f1:.4f} P={p:.4f} R={r:.4f} tp={tp} fp={fp} fn={fn} "
                f"pred_occ={int(keep.sum())} "
                f"pos_sig[mean={pos_mean:.3f} p90={pos_q[2]:.3f}] neg_sig[mean={neg_mean:.3f} p90={neg_q[2]:.3f}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
