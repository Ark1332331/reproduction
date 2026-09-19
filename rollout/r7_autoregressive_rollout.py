"""R7: paper-faithful temporal feedback before 4D sparse-network training.

At time ``t`` the paper does not feed a previous raw camera scan back into the
network.  It transforms the previous *network estimate* into the current robot
frame, labels it ``k=1``, and concatenates it with the current measurement
labelled ``k=0``.  This module makes that contract explicit and testable.

The returned NumPy voxel representation is deliberately detached from any
PyTorch graph.  The paper reports that propagating gradients across the twelve
rollout steps brought no reconstruction benefit.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json

import numpy as np
import torch

from representation.r1_data_representation import (
    VoxelRepresentation,
    align_previous_points_to_current,
    points_to_voxel_representation,
)
from configs.paper_config import (
    PAPER_FINAL_LEARNING_RATE,
    PAPER_GRID_SIZE,
    PAPER_INITIAL_LEARNING_RATE,
    PAPER_PRUNING_ALPHA,
    PAPER_VOXEL_SIZE_M,
)
from models.r5_sparse_input import batch_voxel_representations, make_sparse_tensor
from losses.r5_sparse_loss import completion_loss, downsample_occupancy_target, multiscale_likelihood_loss
from evaluation.r5_sparse_evaluation import (
    current_frame_merge_baseline,
    current_frame_merge_baseline_with_offsets,
    current_frame_prediction_coordinates,
    current_frame_prediction_with_offsets,
    height_metrics,
    occupancy_metrics,
)
from models.r5_sparse_model import EmptyPruningError


@dataclass(frozen=True)
class TemporalTerrainStep:
    """One robot-relative measurement/target pair and pose delta from ``t-1``."""

    current_measurement: np.ndarray
    target_points: np.ndarray
    previous_to_current_translation: tuple[float, float, float]
    previous_to_current_yaw: float


@dataclass(frozen=True)
class AutoregressiveRollout:
    """Inputs and detached estimates created across one temporal sequence."""

    inputs: tuple[VoxelRepresentation, ...]
    estimates: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TemporalTrainingRun:
    """Measured loss values from detached recurrent training on one trajectory."""

    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    learning_rate_history: tuple[float, ...]


def load_isaaclab_temporal_trajectory(path: str) -> tuple[TemporalTerrainStep, ...]:
    """Load the explicit 12-step `.npz` contract from the R7 Isaac Lab capture."""

    with np.load(path, allow_pickle=False) as capture:
        if "metadata_json" not in capture:
            raise ValueError("trajectory is missing metadata_json")
        metadata = json.loads(str(capture["metadata_json"].item()))
        step_count = metadata.get("trajectory_steps")
        if not isinstance(step_count, int) or step_count <= 1:
            raise ValueError("trajectory metadata must contain trajectory_steps greater than one")
        if metadata.get("coordinate_frame") != "robot_centric_local_map":
            raise ValueError("trajectory must use robot_centric_local_map coordinates")
        steps: list[TemporalTerrainStep] = []
        for index in range(step_count):
            suffix = f"{index:02d}"
            names = (f"measurement_{suffix}", f"target_{suffix}", f"translation_{suffix}")
            missing = [name for name in names if name not in capture]
            if missing:
                raise ValueError(f"trajectory is missing arrays: {', '.join(missing)}")
            translation = np.asarray(capture[f"translation_{suffix}"], dtype=float)
            if translation.shape != (3,):
                raise ValueError(f"translation_{suffix} must contain three values")
            steps.append(
                TemporalTerrainStep(
                    current_measurement=_points(capture[f"measurement_{suffix}"], f"measurement_{suffix}"),
                    target_points=_points(capture[f"target_{suffix}"], f"target_{suffix}"),
                    previous_to_current_translation=tuple(float(value) for value in translation),
                    previous_to_current_yaw=float(capture[f"yaw_{suffix}"].item()) if f"yaw_{suffix}" in capture else 0.0,
                )
            )
    return tuple(steps)


def make_autoregressive_voxels(
    current_measurement: np.ndarray,
    previous_estimate: np.ndarray,
    previous_to_current_translation: tuple[float, float, float],
    previous_to_current_yaw: float,
    voxel_size: float = PAPER_VOXEL_SIZE_M,
    grid_size: int = PAPER_GRID_SIZE,
    rotation_center: tuple[float, float, float] | None = None,
) -> VoxelRepresentation:
    """Build the paper's one-step `[x, y, z, k]` input representation.

    ``current_measurement`` is the actual depth-camera observation at ``t``.
    ``previous_estimate`` is a decoded network output from ``t-1``, not an old
    measurement.  After pose alignment it becomes the R1 ``previous_points``
    argument, which assigns its cells ``k=1``.
    """

    current = _points(current_measurement, "current_measurement")
    previous = _points(previous_estimate, "previous_estimate")
    aligned_estimate = align_previous_points_to_current(
        previous,
        translation=previous_to_current_translation,
        yaw=previous_to_current_yaw,
        rotation_center=(
            rotation_center
            if rotation_center is not None
            else (voxel_size * grid_size / 2, voxel_size * grid_size / 2, voxel_size * grid_size / 2)
        ),
    )
    return points_to_voxel_representation(
        current_points=current,
        previous_points=aligned_estimate,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )


def make_autoregressive_voxels_cached_current(
    current_representation: VoxelRepresentation,
    previous_estimate: np.ndarray,
    previous_to_current_translation: tuple[float, float, float],
    previous_to_current_yaw: float,
    voxel_size: float = PAPER_VOXEL_SIZE_M,
    grid_size: int = PAPER_GRID_SIZE,
    rotation_center: tuple[float, float, float] | None = None,
) -> VoxelRepresentation:
    """Build an input while reusing a pre-voxelized current measurement."""
    previous = _points(previous_estimate, "previous_estimate")
    aligned_estimate = align_previous_points_to_current(
        previous,
        translation=previous_to_current_translation,
        yaw=previous_to_current_yaw,
        rotation_center=(
            rotation_center
            if rotation_center is not None
            else (voxel_size * grid_size / 2, voxel_size * grid_size / 2, voxel_size * grid_size / 2)
        ),
    )
    previous_representation = points_to_voxel_representation(
        current_points=np.empty((0, 3), dtype=float),
        previous_points=aligned_estimate,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    return VoxelRepresentation(
        coords=np.vstack([current_representation.coords, previous_representation.coords]),
        features=np.vstack([current_representation.features, previous_representation.features]),
        dropped_count=current_representation.dropped_count + previous_representation.dropped_count,
    )


def rollout_without_temporal_gradients(
    steps: Sequence[TemporalTerrainStep],
    predict_points: Callable[[VoxelRepresentation], np.ndarray],
    voxel_size: float = PAPER_VOXEL_SIZE_M,
    grid_size: int = PAPER_GRID_SIZE,
) -> AutoregressiveRollout:
    """Apply a predictor sequentially and feed each detached estimate forward.

    ``predict_points`` is intentionally a point-cloud boundary: model tensors
    must be decoded to ordinary coordinates before the next pose transform.
    This enforces the paper's no-backpropagation-through-time training choice.
    """

    if not steps:
        raise ValueError("steps must not be empty")
    previous_estimate = np.empty((0, 3), dtype=float)
    inputs: list[VoxelRepresentation] = []
    estimates: list[np.ndarray] = []
    for step in steps:
        representation = make_autoregressive_voxels(
            step.current_measurement,
            previous_estimate,
            step.previous_to_current_translation,
            step.previous_to_current_yaw,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )
        estimate = _points(predict_points(representation), "predict_points result").copy()
        inputs.append(representation)
        estimates.append(estimate)
        previous_estimate = estimate
    return AutoregressiveRollout(inputs=tuple(inputs), estimates=tuple(estimates))


def sparse_prediction_to_current_points(
    coordinates: torch.Tensor,
    likelihood_logits: torch.Tensor,
    offsets: torch.Tensor,
    alpha: float,
    voxel_size: float,
    logit_offset: float = 0.0,
) -> np.ndarray:
    """Decode kept current-frame sparse predictions into ordinary 3D points."""

    if not 0.0 <= alpha <= 1.0 or voxel_size <= 0:
        raise ValueError("alpha must be in [0, 1] and voxel_size must be positive")
    if coordinates.ndim != 2 or coordinates.shape[1] != 5:
        raise ValueError("coordinates must have shape K x 5 [batch, x, y, z, k]")
    if likelihood_logits.shape != (len(coordinates), 1) or offsets.shape != (len(coordinates), 3):
        raise ValueError("likelihood_logits and offsets must have one row per coordinate")
    current_mask = coordinates[:, 4] == 0
    keep_mask = current_mask & (torch.sigmoid(likelihood_logits[:, 0] + logit_offset) >= alpha)
    current_coordinates = coordinates[keep_mask, 1:4].detach().cpu().numpy().astype(float)
    current_offsets = offsets[keep_mask].detach().cpu().numpy()
    return (current_coordinates + current_offsets) * voxel_size


def make_sparse_model_predictor(model, alpha: float, voxel_size: float) -> Callable[[VoxelRepresentation], np.ndarray]:
    """Adapt the real 4D network to the rollout's detached point-cloud boundary."""

    device = next(model.parameters()).device
    model.eval()

    def predict(representation: VoxelRepresentation) -> np.ndarray:
        sparse_input = make_sparse_tensor(batch_voxel_representations([representation]), device=str(device))
        with torch.no_grad():
            prediction = model(sparse_input, alpha=alpha)
        return sparse_prediction_to_current_points(
            prediction.candidates.C,
            prediction.occupancy_logits.F,
            prediction.position_offsets.F,
            alpha=alpha,
            voxel_size=voxel_size,
        )

    return predict


def train_detached_rollout(
    model,
    trajectories: Sequence[Sequence[TemporalTerrainStep]],
    steps: int,
    voxel_size: float = 0.05,
    grid_size: int = 64,
    learning_rate: float = PAPER_INITIAL_LEARNING_RATE,
    final_learning_rate: float = PAPER_FINAL_LEARNING_RATE,
    pruning_alpha: float = PAPER_PRUNING_ALPHA,
    feedback_alpha: float = PAPER_PRUNING_ALPHA,
) -> TemporalTrainingRun:
    """Train with per-step losses but detached model-feedback between time steps.

    The optimizer update is over all time-step losses of every supplied
    trajectory.  The next input is computed by a second, no-grad forward pass
    without target guard, so future inputs cannot accidentally contain ground
    truth through training-time pruning protection.
    """

    if not trajectories or any(not trajectory for trajectory in trajectories):
        raise ValueError("trajectories must contain non-empty sequences")
    if steps <= 0 or learning_rate <= 0 or final_learning_rate <= 0:
        raise ValueError("steps and learning rates must be positive")
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma = 1.0 if steps == 1 else (final_learning_rate / learning_rate) ** (1 / (steps - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_history: list[float] = []
    learning_rate_history: list[float] = []
    for _ in range(steps):
        model.train()
        learning_rate_history.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.zero_grad()
        step_loss_sum = 0.0
        frame_count = 0
        for trajectory in trajectories:
            previous_estimate = np.empty((0, 3), dtype=float)
            for step in trajectory:
                input_representation = make_autoregressive_voxels(
                    step.current_measurement,
                    previous_estimate,
                    step.previous_to_current_translation,
                    step.previous_to_current_yaw,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                )
                target_representation = points_to_voxel_representation(
                    step.target_points,
                    np.empty((0, 3), dtype=float),
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                )
                input_tensor = make_sparse_tensor(
                    batch_voxel_representations([input_representation]), device=str(device)
                )
                target_batch = batch_voxel_representations([target_representation])
                target_scales = tuple(
                    torch.from_numpy(downsample_occupancy_target(target_batch, factor).coordinates)
                    for factor in (8, 4, 2, 1)
                )
                try:
                    prediction = model(
                        input_tensor,
                        alpha=pruning_alpha,
                        training_target_coordinates=target_scales,
                    )
                except EmptyPruningError:
                    # this frame pruned every candidate even with the target guard;
                    # skip it (no loss contribution, previous estimate stays)
                    continue
                final_loss = completion_loss(
                    prediction.candidates.C,
                    prediction.occupancy_logits.F,
                    prediction.position_offsets.F,
                    target_batch,
                ).total
                loss_total = final_loss + multiscale_likelihood_loss(prediction.decoder_likelihoods, target_batch)
                # frame-wise gradient accumulation: the peak GPU memory stays at one
                # frame's computation graph instead of all frames of all trajectories
                # (the 20-trajectory paper distribution would not fit otherwise)
                loss_total.backward()
                step_loss_sum += float(loss_total.detach())
                frame_count += 1
                with torch.no_grad():
                    try:
                        feedback = model(input_tensor, alpha=feedback_alpha)
                        previous_estimate = sparse_prediction_to_current_points(
                            feedback.candidates.C,
                            feedback.occupancy_logits.F,
                            feedback.position_offsets.F,
                            alpha=feedback_alpha,
                            voxel_size=voxel_size,
                        )
                    except EmptyPruningError:
                        # Early in training, alpha pruning may produce a valid
                        # empty reconstruction.  Preserve detached temporal
                        # semantics by feeding an empty history rather than
                        # leaking targets or aborting the rollout.
                        previous_estimate = np.empty((0, 3), dtype=float)
        optimizer.step()
        scheduler.step()
        loss_history.append(step_loss_sum / max(frame_count, 1))
    return TemporalTrainingRun(loss_history[0], loss_history[-1], tuple(loss_history), tuple(learning_rate_history))


def evaluate_detached_rollout(
    model,
    trajectory: Sequence[TemporalTerrainStep],
    voxel_size: float = 0.05,
    grid_size: int = 64,
    alpha: float = 0.5,
    prune_internal: bool = True,
    feedback_alpha: float | None = None,
    use_history: bool = True,
    likelihood_logit_offset: float = 0.0,
) -> dict:
    """Evaluate recurrent predictions and a current-measurement-only baseline.

    ``alpha`` selects this frame's final occupied voxels (for P/R/F1).  With
    ``prune_internal`` enabled it also prunes intermediate decoder candidates.

    ``feedback_alpha`` separately selects which predicted points become the next
    frame's ``previous_estimate`` (k=1 history). It defaults to ``alpha`` for
    backward compatibility; during evaluation an alpha sweep must keep the
    history fixed (feedback_alpha constant) so the sweep does not silently change
    the later frames' inputs.

    ``use_history=False`` is an ablation: every frame receives the current depth
    measurement but an empty k=1 input. It isolates the contribution of the
    recurrent estimate without changing the model or the final metric threshold.
    """

    if not trajectory:
        raise ValueError("trajectory must not be empty")
    if feedback_alpha is None:
        feedback_alpha = alpha
    device = next(model.parameters()).device
    model.eval()
    previous_estimate = np.empty((0, 3), dtype=float)
    baseline_occupancy = []
    prediction_occupancy = []
    baseline_height = []
    prediction_height = []
    for step in trajectory:
        target_representation = points_to_voxel_representation(
            step.target_points, np.empty((0, 3), dtype=float), voxel_size=voxel_size, grid_size=grid_size
        )
        target_batch = batch_voxel_representations([target_representation])
        measurement_representation = points_to_voxel_representation(
            step.current_measurement, np.empty((0, 3), dtype=float), voxel_size=voxel_size, grid_size=grid_size
        )
        measurement_batch = batch_voxel_representations([measurement_representation])
        baseline_coordinates = current_frame_merge_baseline(measurement_batch)
        baseline_with_offsets = current_frame_merge_baseline_with_offsets(measurement_batch)
        baseline_occupancy.append(occupancy_metrics(baseline_coordinates, target_batch))
        baseline_height.append(height_metrics(*baseline_with_offsets, target_batch, voxel_size=voxel_size))
        input_representation = make_autoregressive_voxels(
            step.current_measurement,
            previous_estimate if use_history else np.empty((0, 3), dtype=float),
            step.previous_to_current_translation,
            step.previous_to_current_yaw,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )
        input_tensor = make_sparse_tensor(batch_voxel_representations([input_representation]), device=str(device))
        try:
            with torch.no_grad():
                prediction = model(input_tensor, alpha=alpha if prune_internal else None)
            coordinates = current_frame_prediction_coordinates(
                prediction.candidates.C, prediction.occupancy_logits.F, alpha,
                logit_offset=likelihood_logit_offset,
            )
            coordinates_with_offsets, offsets = current_frame_prediction_with_offsets(
                prediction.candidates.C,
                prediction.occupancy_logits.F,
                prediction.position_offsets.F,
                alpha,
                logit_offset=likelihood_logit_offset,
            )
            previous_estimate = sparse_prediction_to_current_points(
                prediction.candidates.C,
                prediction.occupancy_logits.F,
                prediction.position_offsets.F,
                alpha=feedback_alpha,
                voxel_size=voxel_size,
                logit_offset=likelihood_logit_offset,
            )
        except EmptyPruningError:
            coordinates = np.empty((0, 5), dtype=np.int32)
            coordinates_with_offsets = np.empty((0, 5), dtype=np.int32)
            offsets = np.empty((0, 3), dtype=float)
            previous_estimate = np.empty((0, 3), dtype=float)
        prediction_occupancy.append(occupancy_metrics(coordinates, target_batch))
        prediction_height.append(height_metrics(coordinates_with_offsets, offsets, target_batch, voxel_size=voxel_size))
    return {
        "frame_count": len(trajectory),
        "use_history": use_history,
        "current_measurement_baseline": _summarize_metrics(baseline_occupancy, baseline_height),
        "autoregressive_model": _summarize_metrics(prediction_occupancy, prediction_height),
    }


def _summarize_metrics(occupancy_values, height_values) -> dict:
    """Return paper-style per-frame means plus micro aggregates.

    The paper reports mean precision/recall/F1, while the old implementation
    only exposed a global count-based (micro) aggregate.  Keep the old keys as
    aliases for compatibility, but make both aggregation modes explicit. Height
    error is reported over matched XY cells and now includes its coverage.
    """
    true_positive = sum(value.true_positive for value in occupancy_values)
    false_positive = sum(value.false_positive for value in occupancy_values)
    false_negative = sum(value.false_negative for value in occupancy_values)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = sum(value.matched_cell_count for value in height_values)
    target = sum(value.target_cell_count for value in height_values)
    finite = [value for value in height_values if np.isfinite(value.mean_absolute_error)]
    height_mae = (
        sum(value.mean_absolute_error * value.matched_cell_count for value in finite) / matched if matched else float("inf")
    )
    micro_occupancy = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    micro_height = {
        "matched_cell_count": matched,
        "target_cell_count": target,
        "coverage": matched / target if target else 0.0,
        "mean_absolute_error": height_mae,
    }

    def _finite_mean(values: list[float]) -> float:
        finite_values = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite_values)) if finite_values else float("inf")

    macro_occupancy = {
        "precision": _finite_mean([value.precision for value in occupancy_values]),
        "recall": _finite_mean([value.recall for value in occupancy_values]),
        "f1": _finite_mean([value.f1 for value in occupancy_values]),
    }
    macro_height = {
        "evaluated_frame_count": len(finite),
        "frame_count": len(height_values),
        "matched_cell_count": matched,
        "target_cell_count": target,
        "coverage": matched / target if target else 0.0,
        "mean_absolute_error": _finite_mean([value.mean_absolute_error for value in height_values]),
    }
    return {
        "aggregation": {
            "paper_primary": "macro_per_frame",
            "micro": "global_count_aggregate",
            "height": "matched_xy_cell_weighted_for_micro; per_frame_mean_for_macro",
        },
        "macro": {"occupancy": macro_occupancy, "height": macro_height},
        "micro": {"occupancy": micro_occupancy, "height": micro_height},
        # Backward-compatible aliases. New reports should read macro explicitly.
        "occupancy": micro_occupancy,
        "height": micro_height,
    }


def _points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape N x 3")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array
