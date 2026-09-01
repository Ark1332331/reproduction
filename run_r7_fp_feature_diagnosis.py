"""P3-b diagnostics: FP spatial distribution + frozen-feature separability.

Trains one frame with pos_weight=1 (the working ablation), then:
  1. partitions final candidates into on-target / 1-voxel-adjacent empty /
     farther empty, reporting counts, mean sigmoid and predicted-positive rate;
  2. hooks the final decoder feature (dec0 output, before the 1x1 head) and fits
     a linear classifier on frozen features to separate positive vs negative.

The purpose is to ARCHIVE the "do not change the head" evidence, not to guess at
structure: if frozen features are linearly separable, the head/backbone already
carry enough information and the earlier failures came from the loss weighting.

Usage:

    python run_r7_fp_feature_diagnosis.py --data-dir ../reproduction/data --steps 200
"""

import argparse
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
from r5_sparse_loss import completion_loss


def _neighbour_keys(target_keys):
    neigh = set()
    for (b, x, y, z, k) in target_keys:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neigh.add((b, x + dx, y + dy, z + dz, k))
    return neigh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
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
    for it in range(args.steps):
        model.train()
        optimizer.zero_grad()
        pred = model(input_tensor, alpha=None)
        loss = completion_loss(
            pred.candidates.C, pred.occupancy_logits.F, pred.position_offsets.F, target_batch,
            occupancy_pos_weight=1,
        )
        loss.total.backward()
        optimizer.step()

    # --- hook dec0 output (feature before the 1x1 occupancy head) ---
    captured = {}
    def hook(module, inp, out):
        captured["F"] = out.F.detach()

    model.dec0.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        pred_full = model(input_tensor, alpha=None)
    feats = captured["F"].cpu().numpy()  # aligned with candidates before k0 filter? no: dec0 is before the k0 filter
    probs = torch.sigmoid(pred_full.occupancy_logits.F[:, 0]).detach().cpu().numpy()
    coords = pred_full.candidates.C.detach().cpu().numpy()
    keys = [tuple(int(v) for v in r) for r in coords]
    on = np.array([k in target_keys for k in keys])
    neigh = _neighbour_keys(target_keys)
    near = np.array([k in neigh and not on[i] for i, k in enumerate(keys)])
    far = ~on & ~near

    groups = {"on_target": on, "adjacent_empty": near, "far_empty": far}
    for name, mask in groups.items():
        cnt = int(mask.sum())
        mean_sig = float(probs[mask].mean()) if cnt else 0.0
        pos_rate = float((probs[mask] >= 0.5).mean()) if cnt else 0.0
        print(f"[FP-diag] {name:15s} count={cnt:6d} mean_sig={mean_sig:.4f} predicted_pos_rate={pos_rate:.4f}", flush=True)

    # --- frozen-feature linear separability ---
    # dec0 features are aligned to the candidates BEFORE the k0 filter; the k0
    # filter drops k=1 rows from candidates. Recompute alignment via coordinates:
    # simpler: re-run forward and use the same coordinate order for feats and probs
    # (dec0 out.F has one row per candidate row at that stage). Since the head is
    # 1x1, its output row order equals dec0 row order; the k0 filter then drops
    # k!=0 rows. To keep it simple, fit the linear probe on dec0 features vs the
    # labels computed on the SAME unfiltered rows.
    # We re-derive labels on dec0 coordinates by re-running once more with a hook
    # that also captures coordinates.
    captured2 = {}
    def hook2(module, inp, out):
        captured2["F"] = out.F.detach()
        captured2["C"] = out.C.detach()

    model.dec0.register_forward_hook(hook2)
    with torch.no_grad():
        model(input_tensor, alpha=None)
    feats2 = captured2["F"].cpu().numpy()
    coords2 = captured2["C"].cpu().numpy()
    keys2 = [tuple(int(v) for v in r) for r in coords2]
    labels2 = np.array([k in target_keys for k in keys2])
    # linear probe (logistic regression via sklearn-free least squares on labels)
    X = feats2
    y = labels2.astype(np.float32)
    if len(np.unique(y)) < 2:
        print("[feature-sep] degenerate: single class", flush=True)
    else:
        # closed-form linear classifier: L2-regularized least squares then threshold
        Xt = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        reg = 1e-3
        w = np.linalg.solve(Xt.T @ Xt + reg * np.eye(Xt.shape[1]), Xt.T @ y)
        scores = Xt @ w
        acc = float(((scores >= 0.5) == (y >= 0.5)).mean())
        pos = scores[y >= 0.5]
        neg = scores[y < 0.5]
        sep = float((pos.mean() - neg.mean()) / max(pos.std() + neg.std(), 1e-9))
        print(f"[feature-sep] linear probe acc={acc:.4f} pos_mean={pos.mean():.4f} "
              f"neg_mean={neg.mean():.4f} separation_z={sep:.3f} (n_pos={int(y.sum())} n_neg={int((~y.astype(bool)).sum())})", flush=True)


if __name__ == "__main__":
    main()
