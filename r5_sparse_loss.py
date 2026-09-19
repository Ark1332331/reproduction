"""R5: paper-style occupancy and sub-voxel losses for sparse candidates.

Call path:
    candidate coordinates + model's two feature heads + current-frame target
    -> occupied/not-occupied labels at candidate locations
    -> BCE occupancy loss + mean 3D offset distance on occupied locations

The paper's output is the current estimate P_hat_t. Therefore this module's
target uses only current-frame ground truth (k=0); k=1 is an input-memory label.
This is an implementation inference from Fig. 2 and the training description,
and remains explicit until we have author code or a more detailed appendix.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from r5_sparse_input import SparseVoxelBatch


@dataclass(frozen=True)
class CompletionLoss:
    """Loss values plus the labels that made the calculation inspectable."""

    total: torch.Tensor
    occupancy_loss: torch.Tensor
    position_loss: torch.Tensor
    occupancy_targets: torch.Tensor
    positive_count: int


def downsample_occupancy_target(target: SparseVoxelBatch, spatial_factor: int) -> SparseVoxelBatch:
    """Make the occupied-voxel target for one coarser decoder scale.

    MinkowskiEngine keeps sparse coordinates in the original voxel coordinate
    system and records a coarser resolution in ``tensor_stride``.  A stride-8
    decoder coordinate is therefore ``(..., 48, 32, 0, k)``, not
    ``(..., 6, 4, 0, k)``.  Snap each occupied fine target to the matching
    stride grid instead of changing its coordinate unit.  Batch and time remain
    unchanged; features are not used by likelihood BCE at intermediate levels.
    """
    _validate_current_target(target)
    if spatial_factor <= 0:
        raise ValueError("spatial_factor must be positive")
    coordinates = np.asarray(target.coordinates, dtype=np.int32).copy()
    coordinates[:, 1:4] = (coordinates[:, 1:4] // spatial_factor) * spatial_factor
    coordinates = np.unique(coordinates, axis=0)
    return SparseVoxelBatch(
        coordinates=coordinates,
        features=np.zeros((len(coordinates), 3), dtype=np.float32),
    )


def multiscale_likelihood_loss(
    likelihoods: tuple, target: SparseVoxelBatch, occupancy_pos_weight: float | None = None
) -> torch.Tensor:
    """Average BCE supervision for the intermediate decoder likelihood tensors.

    Supervises strides 8/4/2 (the three decoder layers feeding the next finer
    scale). The final occupancy head l0 is excluded: it is supervised once by
    completion_loss, so giving it an auxiliary weighted/unweighted BCE as well
    would pull the same head toward two conflicting objectives.
    """
    if len(likelihoods) != 3:
        raise ValueError("three intermediate decoder likelihood tensors are required")
    losses = []
    for likelihood, factor in zip(likelihoods, (8, 4, 2)):
        coarse_target = downsample_occupancy_target(target, factor)
        dummy_offsets = torch.zeros((len(likelihood.C), 3), dtype=likelihood.F.dtype, device=likelihood.F.device)
        losses.append(
            completion_loss(
                likelihood.C,
                likelihood.F,
                dummy_offsets,
                coarse_target,
                position_weight=0.0,
                occupancy_pos_weight=occupancy_pos_weight,
            ).occupancy_loss
        )
    return torch.stack(losses).mean()


def completion_loss(
    candidate_coordinates: torch.Tensor,
    occupancy_logits: torch.Tensor,
    position_offsets: torch.Tensor,
    target: SparseVoxelBatch,
    position_weight: float = 1.0,
    occupancy_pos_weight: float | None = None,
    occupancy_loss_weight: float = 1.0,
) -> CompletionLoss:
    """Match sparse candidates to current ground truth and compute both losses.

    ``occupancy_pos_weight`` reweights the occupancy BCE so positive (occupied)
    voxels are not drowned by the empty majority. The special value "auto" uses
    the per-batch negative/positive ratio (clipped to [1, 500]); with generative
    kernel 3 the positive share drops to ~0.5% (800k candidates), where a fixed
    weight like 20 is far too small, which the k3 probes confirmed.
    """
    _validate_prediction_shapes(candidate_coordinates, occupancy_logits, position_offsets)
    _validate_current_target(target)
    if position_weight < 0:
        raise ValueError("position_weight must be non-negative")
    if occupancy_loss_weight < 0:
        raise ValueError("occupancy_loss_weight must be non-negative")

    target_lookup = {
        tuple(int(value) for value in coordinate): feature
        for coordinate, feature in zip(target.coordinates, target.features)
    }
    candidate_coords_np = candidate_coordinates.detach().cpu().numpy()
    # The decoder must predict the k=0 (current-frame) completion only. With the
    # temporal encoder, k=1 voxels propagate through the decoder and appear among
    # the candidates; matching them against the k=0 target would label them all
    # negative (a false-negative flood that collapsed the positive rate to ~1%).
    # So occupancy/offset supervision is restricted to k=0 candidates.
    candidate_k0_mask = candidate_coords_np[:, 4] == 0
    candidate_keys = [
        tuple(int(value) for value in coordinate) for coordinate in candidate_coords_np
    ]
    occupied = [key in target_lookup for key in candidate_keys]
    occupancy_targets = torch.tensor(
        occupied,
        dtype=occupancy_logits.dtype,
        device=occupancy_logits.device,
    ).unsqueeze(1)
    k0_mask = torch.tensor(candidate_k0_mask, dtype=torch.bool, device=occupancy_logits.device)
    if not bool(k0_mask.any()):
        # no k=0 candidates: nothing to supervise (degenerate; return zero losses)
        zero = occupancy_logits.sum() * 0.0
        return CompletionLoss(total=zero, occupancy_loss=zero, position_loss=zero, occupancy_targets=occupancy_targets, positive_count=0)
    supervised_targets = occupancy_targets[k0_mask]
    supervised_logits = occupancy_logits[k0_mask]
    # imbalance correction: with ~2-3% occupied voxels the unweighted BCE is
    # minimized by predicting everything empty, which the R7 paper-distribution
    # experiments confirmed empirically; pos_weight rescales positive gradients
    occupancy_bce_kwargs = {}
    if occupancy_pos_weight is not None:
        if occupancy_pos_weight == "auto":
            positive_count = int(supervised_targets.sum().item())
            total_count = len(supervised_targets)
            ratio = (total_count - positive_count) / max(positive_count, 1)
            effective_weight = float(min(max(ratio, 1.0), 500.0))
        else:
            effective_weight = float(occupancy_pos_weight)
        occupancy_bce_kwargs["pos_weight"] = torch.tensor(
            [effective_weight], dtype=occupancy_logits.dtype, device=occupancy_logits.device
        )
    occupancy = occupancy_loss_weight * F.binary_cross_entropy_with_logits(
        supervised_logits, supervised_targets, **occupancy_bce_kwargs
    )

    positive_mask = supervised_targets[:, 0].bool()
    positive_count = int(positive_mask.sum().item())
    if positive_count == 0:
        position = position_offsets.sum() * 0.0
    else:
        # expected offsets and predicted offsets must both be restricted to the
        # k=0 candidates; positive_mask is indexed into the k0 subset
        expected_offsets_array = np.asarray(
            [
                target_lookup[key]
                for key, is_occupied, is_k0 in zip(candidate_keys, occupied, candidate_k0_mask)
                if is_k0 and is_occupied
            ],
            dtype=np.float32,
        )
        expected_offsets = torch.as_tensor(
            expected_offsets_array,
            dtype=position_offsets.dtype,
            device=position_offsets.device,
        )
        position = torch.linalg.vector_norm(
            position_offsets[k0_mask][positive_mask] - expected_offsets,
            dim=1,
        ).mean()

    total = occupancy + position_weight * position
    return CompletionLoss(
        total=total,
        occupancy_loss=occupancy,
        position_loss=position,
        occupancy_targets=occupancy_targets,
        positive_count=positive_count,
    )


def _validate_prediction_shapes(
    candidate_coordinates: torch.Tensor,
    occupancy_logits: torch.Tensor,
    position_offsets: torch.Tensor,
) -> None:
    candidate_count = len(candidate_coordinates)
    if candidate_coordinates.ndim != 2 or candidate_coordinates.shape[1] != 5:
        raise ValueError("candidate_coordinates must have shape K x 5 [batch, x, y, z, k]")
    if occupancy_logits.shape != (candidate_count, 1):
        raise ValueError("occupancy_logits must have shape K x 1")
    if position_offsets.shape != (candidate_count, 3):
        raise ValueError("position_offsets must have shape K x 3")


def _validate_current_target(target: SparseVoxelBatch) -> None:
    coordinates = np.asarray(target.coordinates)
    features = np.asarray(target.features)
    if coordinates.ndim != 2 or coordinates.shape[1] != 5:
        raise ValueError("target coordinates must have shape K x 5")
    if features.ndim != 2 or features.shape != (len(coordinates), 3):
        raise ValueError("target features must have shape K x 3")
    if not np.all(coordinates[:, 4] == 0):
        raise ValueError("current-frame reconstruction targets must use k=0")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("target coordinates must be unique")
