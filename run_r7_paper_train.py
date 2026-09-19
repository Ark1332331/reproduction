"""Checkpointed training for the R7 paper-distribution experiment.

Trains the four-level 4D completion model on all trajectories under --train-dir
with detached autoregressive feedback (same per-frame gradient accumulation as
r7_autoregressive_rollout.train_detached_rollout), saving a checkpoint every
--save-every steps so training can resume after interruptions.

Usage (GPU, nsr-me-cu130-t291 env):

    python run_r7_paper_train.py --train-dir ../reproduction/data \
        --seeds 0 1 2 3 --steps 200 \
        --checkpoint r7_paper_train.pt [--resume]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from r5_sparse_model import EmptyPruningError, FourLevel4DCompletionModel
from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from r5_sparse_loss import completion_loss, downsample_occupancy_target, multiscale_likelihood_loss
from r7_autoregressive_rollout import (
    load_isaaclab_temporal_trajectory,
    make_autoregressive_voxels,
    make_autoregressive_voxels_cached_current,
    sparse_prediction_to_current_points,
)
from r7_data_augmentation import augment_training_trajectory
from paper_config import (
    IMPLEMENTATION_CHANNELS,
    PAPER_FINAL_LEARNING_RATE,
    PAPER_GRID_SIZE,
    PAPER_INITIAL_LEARNING_RATE,
    PAPER_PRUNING_ALPHA,
    PAPER_TERRAINS,
    PAPER_VOXEL_SIZE_M,
)


def _load_dir(
    directory: Path,
    prefix: str,
    seeds: tuple[int, ...] | None,
    terrains: tuple[str, ...] | None,
    min_trajectory_frames: int = 10,
    max_frames_per_terrain: int | None = None,
):
    trajectories = []
    names = []
    selected_frames = {terrain: 0 for terrain in (terrains or ("stairs", "boxes", "walls", "poles", "corridors"))}
    for terrain in selected_frames:
        for path in sorted(directory.glob(f"{prefix}_{terrain}_*.npz")):
            name = path.stem
            if seeds is not None and not any(f"_s{s}_" in f"{name}_" or name.endswith(f"_s{s}") for s in seeds):
                continue
            trajectory = load_isaaclab_temporal_trajectory(str(path))
            if len(trajectory) < min_trajectory_frames:
                continue
            if max_frames_per_terrain is not None and selected_frames[terrain] + len(trajectory) > max_frames_per_terrain:
                continue
            trajectories.append(trajectory)
            names.append(name)
            selected_frames[terrain] += len(trajectory)
    print(f"[INFO] selected frames by terrain: {selected_frames}", flush=True)
    return trajectories, names


def _split_prediction_points(prediction, batch_size: int, alpha: float, voxel_size: float) -> list[np.ndarray]:
    """Decode a batched sparse prediction into one detached point cloud per item."""
    coordinates = prediction.candidates.C
    outputs = []
    for batch_index in range(batch_size):
        mask = coordinates[:, 0] == batch_index
        outputs.append(
            sparse_prediction_to_current_points(
                coordinates[mask],
                prediction.occupancy_logits.F[mask],
                prediction.position_offsets.F[mask],
                alpha=alpha,
                voxel_size=voxel_size,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--terrains", nargs="*", default=PAPER_TERRAINS)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--min-trajectory-frames", type=int, default=10)
    parser.add_argument("--max-frames-per-terrain", type=int, default=None)
    parser.add_argument("--profile-frames", type=int, default=0, help="Time the first N frames of the first step.")
    parser.add_argument("--microbatch-size", type=int, default=1, help="Number of trajectories per temporal micro-batch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size", type=float, default=PAPER_VOXEL_SIZE_M)
    parser.add_argument("--grid-size", type=int, default=PAPER_GRID_SIZE)
    parser.add_argument("--pruning-alpha", type=float, default=PAPER_PRUNING_ALPHA)
    parser.add_argument(
        "--pruning-warmup-steps", type=int, default=0,
        help="Use alpha=0 (no candidate pruning) for the first N optimizer steps.",
    )
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
        "--position-weight", type=float, default=1.0,
        help="Weight of the final positive-voxel offset loss; 0 isolates occupancy calibration",
    )
    parser.add_argument(
        "--occupancy-loss-weight", type=float, default=1.0,
        help="Scale final occupancy BCE relative to position loss; default preserves the paper-shaped loss",
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
    parser.add_argument(
        "--feedback-alpha", type=float, default=None,
        help="Threshold for feedback history; defaults to the paper pruning alpha.",
    )
    parser.add_argument("--channels", type=int, nargs=4, default=IMPLEMENTATION_CHANNELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=PAPER_INITIAL_LEARNING_RATE)
    parser.add_argument("--final-learning-rate", type=float, default=PAPER_FINAL_LEARNING_RATE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-data-augmentation", action="store_true",
        help="Diagnostic ablation; paper training applies measurement noise and trajectory mirroring.",
    )
    parser.add_argument(
        "--target-guard", action=argparse.BooleanOptionalAction, default=False,
        help="Engineering safeguard: preserve generated candidates that coincide with targets. "
             "Off by default because this is not a published paper detail.",
    )
    parser.add_argument(
        "--strict-paper", action="store_true",
        help="Disable target guard so the run does not use the unpublished training-time candidate safeguard.",
    )
    args = parser.parse_args()

    if args.strict_paper:
        args.target_guard = False
    if args.feedback_alpha is None:
        args.feedback_alpha = args.pruning_alpha

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.min_trajectory_frames < 2 or (args.max_frames_per_terrain is not None and args.max_frames_per_terrain <= 0):
        raise ValueError("invalid trajectory/frame limits")
    if args.position_weight < 0:
        raise ValueError("position-weight must be non-negative")
    if args.occupancy_loss_weight < 0:
        raise ValueError("occupancy-loss-weight must be non-negative")
    if args.profile_frames < 0 or args.pruning_warmup_steps < 0:
        raise ValueError("profile-frames must be non-negative")
    if args.microbatch_size <= 0:
        raise ValueError("microbatch-size must be positive")
    train_trajectories, train_names = _load_dir(
        args.train_dir, "isaac_anymal", None if args.seeds is None else tuple(args.seeds), args.terrains,
        args.min_trajectory_frames, args.max_frames_per_terrain,
    )
    if not train_trajectories:
        raise RuntimeError(f"no training trajectories in {args.train_dir}")
    if not args.disable_data_augmentation:
        augmentation_rng = np.random.default_rng(args.seed)
        train_trajectories = [
            augment_training_trajectory(trajectory, augmentation_rng)
            for trajectory in train_trajectories
        ]
        print("[INFO] enabled Section III-E measurement augmentation and trajectory mirroring", flush=True)
    print(f"[INFO] train: {len(train_trajectories)} trajectories {train_names}", flush=True)
    cached_current = [
        [
            points_to_voxel_representation(
                frame.current_measurement, np.empty((0, 3), dtype=float),
                voxel_size=args.voxel_size, grid_size=args.grid_size,
            )
            for frame in trajectory
        ]
        for trajectory in train_trajectories
    ]
    cached_targets = [
        [
            points_to_voxel_representation(
                frame.target_points, np.empty((0, 3), dtype=float),
                voxel_size=args.voxel_size, grid_size=args.grid_size,
            )
            for frame in trajectory
        ]
        for trajectory in train_trajectories
    ]

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
    profile_totals = {name: 0.0 for name in ("input_voxel", "target_voxel", "sparse_batch", "train_forward", "loss", "backward", "feedback")}
    profile_count = 0

    def _sync_for_profile() -> None:
        if args.profile_frames and str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
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
        step_pruning_alpha = 0.0 if step < args.pruning_warmup_steps else args.pruning_alpha
        optimizer.zero_grad()
        step_loss_sum = 0.0
        final_loss_sum = 0.0
        likelihood_loss_sum = 0.0
        frame_count = 0
        import time as _time

        _t_pre = _time.time()
        if args.microbatch_size == 1:
            trajectory_groups = [[trajectory] for trajectory in train_trajectories]
        else:
            trajectory_groups = [
                train_trajectories[start : start + args.microbatch_size]
                for start in range(0, len(train_trajectories), args.microbatch_size)
            ]
        for group_start, trajectory_group in zip(
            range(0, len(train_trajectories), args.microbatch_size), trajectory_groups
        ):
            if args.microbatch_size == 1:
                frame_iterator = ((0, frame) for frame in trajectory_group[0])
                previous_estimates = [np.empty((0, 3), dtype=float)]
            else:
                previous_estimates = [np.empty((0, 3), dtype=float) for _ in trajectory_group]
                frame_iterator = None
            if args.microbatch_size == 1:
                iterator = frame_iterator
            else:
                iterator = ((frame_index, None) for frame_index in range(max(map(len, trajectory_group))))
            for frame_index, frame in iterator:
                if args.microbatch_size > 1:
                    active = [(index, trajectory_group[index][frame_index]) for index in range(len(trajectory_group)) if frame_index < len(trajectory_group[index])]
                    if not active:
                        continue
                    frames = [item[1] for item in active]
                    active_indices = [item[0] for item in active]
                else:
                    frames = [frame]
                    active_indices = [0]
                profile_this = step == start_step and profile_count < args.profile_frames and len(frames) == 1
                if profile_this:
                    _sync_for_profile()
                    _t_profile = _time.perf_counter()
                input_representations = [
                    make_autoregressive_voxels_cached_current(
                        cached_current[group_start + index][frame_index],
                        previous_estimates[index],
                        current_frame.previous_to_current_translation,
                        current_frame.previous_to_current_yaw,
                        voxel_size=args.voxel_size,
                        grid_size=args.grid_size,
                    )
                    for index, current_frame in zip(active_indices, frames)
                ]
                if profile_this:
                    _sync_for_profile()
                    profile_totals["input_voxel"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                target_representations = [cached_targets[group_start + index][frame_index] for index in active_indices]
                if profile_this:
                    _sync_for_profile()
                    profile_totals["target_voxel"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                input_tensor = make_sparse_tensor(
                    batch_voxel_representations(input_representations), device=str(args.device)
                )
                target_batch = batch_voxel_representations(target_representations)
                target_scales = tuple(
                    torch.from_numpy(downsample_occupancy_target(target_batch, factor).coordinates)
                    for factor in (8, 4, 2, 1)
                )
                if profile_this:
                    _sync_for_profile()
                    profile_totals["sparse_batch"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                prediction = model(
                    input_tensor,
                    alpha=step_pruning_alpha,
                    training_target_coordinates=target_scales if args.target_guard else None,
                )
                if profile_this:
                    _sync_for_profile()
                    profile_totals["train_forward"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                final_loss = completion_loss(
                    prediction.candidates.C,
                    prediction.occupancy_logits.F,
                    prediction.position_offsets.F,
                    target_batch,
                    position_weight=args.position_weight,
                    occupancy_pos_weight=args.occupancy_pos_weight,
                    occupancy_loss_weight=args.occupancy_loss_weight,
                ).total
                likelihood_loss = multiscale_likelihood_loss(
                    prediction.decoder_likelihoods, target_batch, occupancy_pos_weight=args.likelihood_pos_weight
                )
                loss_total = final_loss + args.likelihood_weight * likelihood_loss
                if profile_this:
                    _sync_for_profile()
                    profile_totals["loss"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                loss_total.backward()
                if profile_this:
                    _sync_for_profile()
                    profile_totals["backward"] += _time.perf_counter() - _t_profile
                    _t_profile = _time.perf_counter()
                batch_count = len(frames)
                step_loss_sum += float(loss_total.detach()) * batch_count
                final_loss_sum += float(final_loss.detach()) * batch_count
                likelihood_loss_sum += float(likelihood_loss.detach()) * batch_count
                frame_count += batch_count
                if args.clear_me_cache:
                    # free the backward graph buffers before the no-grad feedback
                    # forward: with 800k candidates the two forwards together peak
                    # past the 8 GB GPU (observed OOM in the up0 generative layer)
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    try:
                        feedback = model(input_tensor, alpha=args.feedback_alpha)
                        estimates = _split_prediction_points(
                            feedback, batch_count, args.feedback_alpha, args.voxel_size
                        )
                    except EmptyPruningError:
                        # An untrained/early model may legitimately predict no
                        # occupied voxels at the feedback threshold.  Treat that
                        # as an empty detached history rather than aborting the
                        # rollout or leaking target points into the next frame.
                        estimates = [np.empty((0, 3), dtype=float) for _ in range(batch_count)]
                    for index, estimate in zip(active_indices, estimates):
                        previous_estimates[index] = estimate
                if profile_this:
                    _sync_for_profile()
                    profile_totals["feedback"] += _time.perf_counter() - _t_profile
                    profile_count += 1
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
        if profile_count:
            averages = ", ".join(f"{name}={value / profile_count:.3f}s" for name, value in profile_totals.items())
            print(f"[profile] frames={profile_count} {averages}", flush=True)
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
                        "schema_version": 3,
                        "channels": list(args.channels),
                        "generative_kernels": kernels,
                        "voxel_size": args.voxel_size,
                        "grid_size": args.grid_size,
                        "pruning_alpha": args.pruning_alpha,
                        "pruning_warmup_steps": args.pruning_warmup_steps,
                        "feedback_alpha": args.feedback_alpha,
                        "occupancy_pos_weight": args.occupancy_pos_weight,
                        "likelihood_weight": args.likelihood_weight,
                        "likelihood_pos_weight": args.likelihood_pos_weight,
                        "position_weight": args.position_weight,
                        "occupancy_loss_weight": args.occupancy_loss_weight,
                        "learning_rate": args.learning_rate,
                        "data_augmentation": not args.disable_data_augmentation,
                        "target_guard": args.target_guard,
                        "strict_paper": args.strict_paper,
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
