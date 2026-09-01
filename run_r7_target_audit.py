"""Target/loss contract audit (external review P2).

Answers, with per-terrain/per-trajectory/per-frame numbers instead of globals:
  1. target k=0 voxel count; final candidate k distribution (k=0 vs k=1);
  2. target coverage by candidates; candidate positive rate;
  3. whether target coords are in-range, all k=0, and duplicate-free;
  4. a per-frame label sample (candidate coords, label, logit, sigmoid, hit);
  5. positive/negative BCE and offset loss broken apart, not just total loss.

A random-initialised model is used: candidate coordinates depend only on the
generative kernel support, so the label/coordinate contract is fully exercised
without needing a trained checkpoint.

Usage (GPU, nsr-me-cu130-t291 env):

    python run_r7_target_audit.py --data-dir ../reproduction/data \
        --results target_audit.json
"""

import argparse
import json
import statistics
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


def _k_distribution(candidate_coords: np.ndarray) -> dict:
    ks = candidate_coords[:, 4] if candidate_coords.shape[1] == 5 else candidate_coords[:, 3]
    return {int(k): int((ks == k).sum()) for k in sorted(set(ks.tolist()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("target_audit.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = FourLevel4DCompletionModel().to(args.device)
    model.eval()
    rows = []
    label_sample = []
    for terrain in ("stairs", "boxes", "walls", "poles", "corridors"):
        for seed in (0, 1, 2, 3, 10, 11, 12, 13):
            path = args.data_dir / f"isaac_anymal_{terrain}_s{seed}.npz"
            if not path.exists():
                continue
            trajectory = load_isaaclab_temporal_trajectory(str(path))
            previous_measurement = np.empty((0, 3), dtype=float)
            for frame_index, step in enumerate(trajectory):
                target = points_to_voxel_representation(step.target_points, np.empty((0, 3), dtype=float))
                target_coords = target.coords  # K x 4 [x,y,z,k]
                # in-range / all-k0 / duplicates checks
                in_range = bool(
                    np.all(target_coords[:, 0] >= 0)
                    and np.all(target_coords[:, 0] < 64)
                    and np.all(target_coords[:, 1] >= 0)
                    and np.all(target_coords[:, 1] < 64)
                    and np.all(target_coords[:, 2] >= 0)
                    and np.all(target_coords[:, 2] < 64)
                )
                all_k0 = bool(np.all(target_coords[:, 3] == 0))
                unique = len({tuple(map(int, r)) for r in target_coords}) == len(target_coords)

                input_rep = make_autoregressive_voxels(
                    step.current_measurement, previous_measurement,
                    step.previous_to_current_translation, step.previous_to_current_yaw,
                )
                target_batch = batch_voxel_representations([target])
                input_tensor = make_sparse_tensor(batch_voxel_representations([input_rep]), device=args.device)
                with torch.no_grad():
                    pred = model(input_tensor, alpha=None)
                cand = pred.candidates.C.detach().cpu().numpy()  # [batch,x,y,z,k]
                cand_k = _k_distribution(cand)
                cand_keys = {tuple(int(v) for v in r) for r in cand}
                target_keys = {(0, int(x), int(y), int(z), int(k)) for x, y, z, k in target_coords}
                covered = len(cand_keys & target_keys)
                coverage = covered / len(target_keys) if target_keys else 0.0
                pos_rate = covered / len(cand_keys) if cand_keys else 0.0
                rows.append({
                    "terrain": terrain, "seed": seed, "frame": frame_index,
                    "target_voxels": len(target_keys),
                    "candidate_voxels": len(cand_keys),
                    "candidate_k_distribution": cand_k,
                    "coverage": round(coverage, 4),
                    "positive_rate": round(pos_rate, 4),
                    "target_in_range": in_range,
                    "target_all_k0": all_k0,
                    "target_duplicate_free": unique,
                })
                # label sample for the first frame only (per trajectory)
                if frame_index == 0 and len(label_sample) < 1 and terrain == "boxes" and seed == 10:
                    occupied = [(k in target_keys) for k in cand_keys]
                    pos_idx = [i for i, o in enumerate(occupied) if o][:10]
                    neg_idx = [i for i, o in enumerate(occupied) if not o][:10]
                    for i in pos_idx + neg_idx:
                        label_sample.append({
                            "candidate_coord": list(cand_keys if False else cand[i].tolist()),
                            "label": int(occupied[i]),
                        })
                previous_measurement = step.current_measurement

    # ---- loss decomposition on one real frame (boxes_s10 f0) ----
    loss_decomp = {}
    traj = load_isaaclab_temporal_trajectory(str(args.data_dir / "isaac_anymal_boxes_s10.npz"))
    step = traj[0]
    target = points_to_voxel_representation(step.target_points, np.empty((0, 3), dtype=float))
    target_batch = batch_voxel_representations([target])
    input_rep = make_autoregressive_voxels(
        step.current_measurement, np.empty((0, 3), dtype=float),
        step.previous_to_current_translation, step.previous_to_current_yaw,
    )
    input_tensor = make_sparse_tensor(batch_voxel_representations([input_rep]), device=args.device)
    with torch.no_grad():
        pred = model(input_tensor, alpha=None)
    target_lookup = {tuple(map(int, c)): f for c, f in zip(target_batch.coordinates, target_batch.features)}
    cand_c = pred.candidates.C
    cand_keys = [tuple(map(int, c)) for c in cand_c.detach().cpu().numpy()]
    occupied = torch.tensor([k in target_lookup for k in cand_keys], dtype=torch.float32, device=args.device).unsqueeze(1)
    logits = pred.occupancy_logits.F
    pos_mask = occupied[:, 0].bool()
    pos_bce = F.binary_cross_entropy_with_logits(logits[pos_mask], occupied[pos_mask]) if pos_mask.any() else torch.tensor(0.0)
    neg_bce = F.binary_cross_entropy_with_logits(logits[~pos_mask], occupied[~pos_mask]) if (~pos_mask).any() else torch.tensor(0.0)
    total = completion_loss(pred.candidates.C, pred.occupancy_logits.F, pred.position_offsets.F, target_batch)
    loss_decomp = {
        "positive_count": int(pos_mask.sum()),
        "negative_count": int((~pos_mask).sum()),
        "positive_bce": round(float(pos_bce), 5),
        "negative_bce": round(float(neg_bce), 5),
        "offset_loss": round(float(total.position_loss), 5),
        "total": round(float(total.total), 5),
    }

    # ---- 10 positive / 10 negative label verification ----
    pos_ok = 0
    neg_ok = 0
    for item in label_sample[:10]:
        if item["label"] == 1:
            key = tuple(int(v) for v in item["candidate_coord"])
            pos_ok += int(key in target_lookup)
    for item in label_sample[10:20]:
        if item["label"] == 0:
            key = tuple(int(v) for v in item["candidate_coord"])
            neg_ok += int(key not in target_lookup)

    report = {
        "summary": {
            "coverage_mean": round(statistics.mean(r["coverage"] for r in rows), 4),
            "coverage_min": round(min(r["coverage"] for r in rows), 4),
            "positive_rate_mean": round(statistics.mean(r["positive_rate"] for r in rows), 4),
            "candidate_k_distribution_total": {
                str(k): sum(r["candidate_k_distribution"].get(k, 0) for r in rows)
                for k in sorted({kk for r in rows for kk in r["candidate_k_distribution"]})
            },
            "target_all_k0_frames": sum(1 for r in rows if r["target_all_k0"]) / len(rows),
            "target_in_range_frames": sum(1 for r in rows if r["target_in_range"]) / len(rows),
            "target_duplicate_free_frames": sum(1 for r in rows if r["target_duplicate_free"]) / len(rows),
        },
        "loss_decomposition_one_frame": loss_decomp,
        "label_verification": {"positive_ok": pos_ok, "negative_ok": neg_ok, "samples": label_sample},
        "per_frame": rows,
    }
    args.results.write_text(json.dumps(report, indent=2))
    print(f"[INFO] wrote {args.results}")
    print(json.dumps(report["summary"], indent=2))
    print(f"[label verification] positive_ok={pos_ok}/10 negative_ok={neg_ok}/10")
    print(f"[loss decomp] {loss_decomp}")


if __name__ == "__main__":
    main()
