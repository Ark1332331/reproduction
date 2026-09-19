"""R5: paper-aligned occupancy evaluation for current-frame sparse voxels.

The paper reports voxel-grid precision, recall, and F1. This module makes all
three compared objects use [batch, x, y, z, k=0] coordinates:
    input observations -> non-learning merge baseline
    model candidates + logits -> thresholded current-frame prediction
    complete current terrain -> ground-truth target
"""

from dataclasses import dataclass

import numpy as np
import torch

from models.r5_sparse_input import SparseVoxelBatch


@dataclass(frozen=True)
class SparseOccupancyMetrics:
    """Counts and rates for occupied current-frame voxel coordinates."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class SparseHeightMetrics:
    """Height error over horizontal cells represented by both point clouds."""

    matched_cell_count: int
    target_cell_count: int
    mean_absolute_error: float


def current_frame_merge_baseline(observed_input: SparseVoxelBatch) -> np.ndarray:
    """Merge current/history observations after both are expressed in current space.

    The input's k=1 records are historical observations. Their spatial locations
    are already pose-aligned by R1, so the baseline re-labels both sources as the
    current output time k=0 and removes duplicate occupied locations.
    """
    coordinates = _as_coordinate_array(observed_input.coordinates, "observed_input")
    merged = coordinates.copy()
    merged[:, 4] = 0
    return np.unique(merged, axis=0).astype(np.int32)


def current_frame_merge_baseline_with_offsets(observed_input: SparseVoxelBatch) -> tuple[np.ndarray, np.ndarray]:
    """Merge observed voxels while retaining one sub-voxel offset per voxel.

    When aligned current/history observations share a voxel their offsets only
    differ due to measurement noise, so the first observation is retained.
    """
    coordinates = _as_coordinate_array(observed_input.coordinates, "observed_input")
    features = np.asarray(observed_input.features, dtype=float)
    if features.shape != (len(coordinates), 3):
        raise ValueError("observed_input.features must have shape K x 3")
    merged = coordinates.copy()
    merged[:, 4] = 0
    _, first_indices = np.unique(merged, axis=0, return_index=True)
    return merged[first_indices].astype(np.int32), features[first_indices]


def current_frame_prediction_coordinates(
    candidate_coordinates: torch.Tensor,
    occupancy_logits: torch.Tensor,
    alpha: float,
    logit_offset: float = 0.0,
) -> np.ndarray:
    """Threshold model likelihoods and discard history-time candidates from output."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if candidate_coordinates.ndim != 2 or candidate_coordinates.shape[1] != 5:
        raise ValueError("candidate_coordinates must have shape K x 5")
    if occupancy_logits.shape != (len(candidate_coordinates), 1):
        raise ValueError("occupancy_logits must have shape K x 1")

    coordinates = candidate_coordinates.detach().cpu().numpy().astype(np.int32)
    likelihoods = torch.sigmoid(occupancy_logits[:, 0] + logit_offset).detach().cpu().numpy()
    keep = (coordinates[:, 4] == 0) & (likelihoods >= alpha)
    return np.unique(coordinates[keep], axis=0).astype(np.int32)


def current_frame_prediction_with_offsets(
    candidate_coordinates: torch.Tensor,
    occupancy_logits: torch.Tensor,
    position_offsets: torch.Tensor,
    alpha: float,
    logit_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select current occupied predictions and retain their 3D offset estimates."""
    if position_offsets.shape != (len(candidate_coordinates), 3):
        raise ValueError("position_offsets must have shape K x 3")
    coordinates = candidate_coordinates.detach().cpu().numpy().astype(np.int32)
    likelihoods = torch.sigmoid(occupancy_logits[:, 0] + logit_offset).detach().cpu().numpy()
    offsets = position_offsets.detach().cpu().numpy()
    keep = (coordinates[:, 4] == 0) & (likelihoods >= alpha)
    return coordinates[keep], offsets[keep]


def occupancy_metrics(
    predicted_coordinates: np.ndarray,
    target: SparseVoxelBatch,
) -> SparseOccupancyMetrics:
    """Compare predicted occupied voxels with complete current-frame target voxels."""
    predicted = _as_coordinate_array(predicted_coordinates, "predicted_coordinates")
    target_coordinates = _as_coordinate_array(target.coordinates, "target.coordinates")
    if not np.all(predicted[:, 4] == 0):
        raise ValueError("predicted coordinates must represent the current frame (k=0)")
    if not np.all(target_coordinates[:, 4] == 0):
        raise ValueError("target coordinates must represent the current frame (k=0)")

    predicted_set = {tuple(int(value) for value in row) for row in predicted}
    target_set = {tuple(int(value) for value in row) for row in target_coordinates}
    true_positive = len(predicted_set & target_set)
    false_positive = len(predicted_set - target_set)
    false_negative = len(target_set - predicted_set)

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return SparseOccupancyMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def height_metrics(
    predicted_coordinates: np.ndarray,
    predicted_offsets: np.ndarray,
    target: SparseVoxelBatch,
    voxel_size: float,
) -> SparseHeightMetrics:
    """Measure surface-height MAE after taking the highest voxel in each XY cell.

    MAE is defined where reconstruction and target both contain a horizontal
    cell. The count of such cells is returned so a sparse output cannot hide
    behind a low error on only a few cells; occupancy F1 remains its companion.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    predicted = _as_coordinate_array(predicted_coordinates, "predicted_coordinates")
    predicted_features = np.asarray(predicted_offsets, dtype=float)
    if predicted_features.shape != (len(predicted), 3):
        raise ValueError("predicted_offsets must have shape K x 3")
    target_coordinates = _as_coordinate_array(target.coordinates, "target.coordinates")
    if not np.all(predicted[:, 4] == 0) or not np.all(target_coordinates[:, 4] == 0):
        raise ValueError("height metrics require current-frame k=0 coordinates")
    predicted_heights = _surface_heights(predicted, predicted_features, voxel_size)
    target_heights = _surface_heights(target_coordinates, target.features, voxel_size)
    shared = predicted_heights.keys() & target_heights.keys()
    if not shared:
        return SparseHeightMetrics(0, len(target_heights), float("inf"))
    errors = [abs(predicted_heights[cell] - target_heights[cell]) for cell in shared]
    return SparseHeightMetrics(len(shared), len(target_heights), float(np.mean(errors)))


def _as_coordinate_array(coordinates: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(coordinates)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(f"{name} must have shape K x 5 [batch, x, y, z, k]")
    return array


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _surface_heights(coordinates: np.ndarray, offsets: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], float]:
    """Map [batch, x, y] to the highest decoded z coordinate."""
    heights: dict[tuple[int, int, int], float] = {}
    for coordinate, offset in zip(coordinates, offsets, strict=True):
        cell = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
        height = float((coordinate[3] + offset[2]) * voxel_size)
        heights[cell] = max(heights.get(cell, float("-inf")), height)
    return heights
