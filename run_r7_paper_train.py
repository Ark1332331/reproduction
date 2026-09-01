"""Checkpointed training for the R7 paper-distribution experiment.

Trains the four-level 4D completion model on all trajectories under --train-dir
with detached autoregressive feedback (same per-frame gradient accumulation as
r7_autoregressive_rollout.train_detached_rollout), saving a checkpoint every
--save-every steps so training can resume after interruptions.

Usage (GPU, nsr-me-cu130-t291 env):

    python run_r7_paper_train.py --train-dir ../reproduction/data \
        --seeds 0 1 2 3 --steps 200 --pruning-alpha 0.0 \
        --checkpoint r7_paper_train.pt [--resume]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from r5_sparse_model import FourLevel4DCompletionModel
from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from r5_sparse_loss import completion_loss, downsample_occupancy_target, multiscale_likelihood_loss
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    make_autoregressive_voxels,
    sparse_prediction_to_current_points,
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
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--terrains", nargs="*", default=("stairs", "boxes", "walls", "poles", "corridors"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--pruning-alpha", type=float, default=0.0)
    parser.add_argument(
        "--occupancy-pos-weight", default=None,
        help="BCE positive-class weight for occupancy; 'auto' uses the per-batch "
        "negative/positive ratio (recommended with kernel-3 candidates); None = unweighted",
    )
    parser.add_argument(
        "--likelihood-weight", type=float, default=1.0,
        help="Weight of the per-layer likelihood BCE (paper uses it; 0 disables for ablation)",
    )
    parser.add_argument(
        "--likelihood-pos-weight", type=float, default=None,
        help="BCE positive-class weight for the per-layer likelihood losses; None = unweighted",
    )
    parser.add_argument(
        "--generative-kernels", type=int, nargs=16, default=None,
        help="Four 4D generative kernels (16 ints) for up3/up2/up1/up0; default (2,2,2,1) x4",
    )
    parser.add_argument(
        "--clear-me-cache", action="store_true",
        help="Clear the MinkowskiEngine global coordinate manager and empty the CUDA "
        "cache after every step (needed with large generative kernels on 8 GB GPUs)",
    )
    parser.add_argument("--feedback-alpha", type=float, default=0.0)
    parser.add_argument("--channels", type=int, nargs=4, default=(4, 8, 12, 16))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--final-learning-rate", type=float, default=0.0001)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_trajectories, train_names = _load_dir(args.train_dir, "isaac_anymal", tuple(args.seeds or ()), args.terrains)
    if not train_trajectories:
        raise RuntimeError(f"no training trajectories in {args.train_dir}")
    print(f"[INFO] train: {len(train_trajectories)} trajectories {train_names}", flush=True)

    kernels = None
    if args.generative_kernels is not None:
        kernels = tuple(tuple(args.generative_kernels[i : i + 4]) for i in range(0, 16, 4))
        print(f"[INFO] generative kernels: {kernels}", flush=True)
    if kernels is not None:
        model = FourLevel4DCompletionModel(channels=tuple(args.channels), generative_kernels=kernels).to(args.device)
    else:
        model = FourLevel4DCompletionModel(channels=tuple(args.channels)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    gamma = 1.0 if args.steps == 1 else (args.final_learning_rate / args.learning_rate) ** (1 / (args.steps - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    start_step = 0
    loss_history = []
    final_loss_history = []
    likelihood_loss_history = []
    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"] + 1
        loss_history = list(state.get("loss_history", []))
        final_loss_history = list(state.get("final_loss_history", []))
        likelihood_loss_history = list(state.get("likelihood_loss_history", []))
        print(f"[INFO] resumed from step {state['step']}", flush=True)

    for step in range(start_step, args.steps):
        model.train()
        optimizer.zero_grad()
        step_loss_sum = 0.0
        final_loss_sum = 0.0
        likelihood_loss_sum = 0.0
        frame_count = 0
        import time as _time

        _t_pre = _time.time()
        for trajectory in train_trajectories:
            previous_estimate = np.empty((0, 3), dtype=float)
            for frame in trajectory:
                input_representation = make_autoregressive_voxels(
                    frame.current_measurement,
                    previous_estimate,
                    frame.previous_to_current_translation,
                    frame.previous_to_current_yaw,
                    voxel_size=args.voxel_size,
                    grid_size=args.grid_size,
                )
                target_representation = points_to_voxel_representation(
                    frame.target_points, np.empty((0, 3), dtype=float),
                    voxel_size=args.voxel_size, grid_size=args.grid_size,
                )
                input_tensor = make_sparse_tensor(
                    batch_voxel_representations([input_representation]), device=str(args.device)
                )
                target_batch = batch_voxel_representations([target_representation])
                target_scales = tuple(
                    torch.from_numpy(downsample_occupancy_target(target_batch, factor).coordinates)
                    for factor in (8, 4, 2, 1)
                )
                prediction = model(
                    input_tensor,
                    alpha=args.pruning_alpha,
                    training_target_coordinates=target_scales,
                )
                final_loss = completion_loss(
                    prediction.candidates.C,
                    prediction.occupancy_logits.F,
                    prediction.position_offsets.F,
                    target_batch,
                    occupancy_pos_weight=args.occupancy_pos_weight,
                ).total
                likelihood_loss = multiscale_likelihood_loss(
                    prediction.decoder_likelihoods, target_batch, occupancy_pos_weight=args.likelihood_pos_weight
                )
                loss_total = final_loss + args.likelihood_weight * likelihood_loss
                loss_total.backward()
                step_loss_sum += float(loss_total.detach())
                final_loss_sum += float(final_loss.detach())
                likelihood_loss_sum += float(likelihood_loss.detach())
                frame_count += 1
                if args.clear_me_cache:
                    # free the backward graph buffers before the no-grad feedback
                    # forward: with 800k candidates the two forwards together peak
                    # past the 8 GB GPU (observed OOM in the up0 generative layer)
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    feedback = model(input_tensor, alpha=args.feedback_alpha)
                    previous_estimate = sparse_prediction_to_current_points(
                        feedback.candidates.C,
                        feedback.occupancy_logits.F,
                        feedback.position_offsets.F,
                        alpha=args.feedback_alpha,
                        voxel_size=args.voxel_size,
                    )
        optimizer.step()
        scheduler.step()
        if args.clear_me_cache:
            # MinkowskiEngine keeps every frame's coordinate maps alive during a
            # step; with large generative kernels the accumulated maps/features
            # blew past the 8 GB GPU (observed OOM at ~7 GB), so clear them after
            # the backward pass of each step. Safe: no tensor from the step is
            # used afterwards (feedback ran under no_grad inside the loop).
            import MinkowskiEngine as _ME

            _ME.clear_global_coordinate_manager()
            torch.cuda.empty_cache()
        mean_loss = step_loss_sum / max(frame_count, 1)
        mean_final_loss = final_loss_sum / max(frame_count, 1)
        mean_likelihood_loss = likelihood_loss_sum / max(frame_count, 1)
        loss_history.append(mean_loss)
        final_loss_history.append(mean_final_loss)
        likelihood_loss_history.append(mean_likelihood_loss)
        _t_fwd = _time.time()
        print(
            f"[step {step:03d}] loss {mean_loss:.4f} final {mean_final_loss:.4f} "
            f"likelihood {mean_likelihood_loss:.4f} lr {scheduler.get_last_lr()[0]:.6f} "
            f"pre {_t_fwd - _t_pre:.1f}s",
            flush=True,
        )
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "loss_history": loss_history,
                    "final_loss_history": final_loss_history,
                    "likelihood_loss_history": likelihood_loss_history,
                    "config": {
                        "schema_version": 2,
                        "channels": list(args.channels),
                        "generative_kernels": kernels,
                        "voxel_size": args.voxel_size,
                        "grid_size": args.grid_size,
                        "pruning_alpha": args.pruning_alpha,
                        "feedback_alpha": args.feedback_alpha,
                        "occupancy_pos_weight": args.occupancy_pos_weight,
                        "likelihood_weight": args.likelihood_weight,
                        "likelihood_pos_weight": args.likelihood_pos_weight,
                        "learning_rate": args.learning_rate,
                        "final_learning_rate": args.final_learning_rate,
                        "requested_steps": args.steps,
                        "seed": args.seed,
                        "clear_me_cache": args.clear_me_cache,
                        "requested_seeds": args.seeds,
                        "requested_terrains": list(args.terrains),
                        "train_trajectories": train_names,
                    },
                },
                args.checkpoint,
            )
            print(f"[save] {args.checkpoint} (step {step})", flush=True)

    print(f"[DONE] trained {args.steps} steps; loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
