"""R3-a: merge raw current/history toy point clouds as a baseline.

Purpose:
    Isolate the completion task before introducing a neural network.
Upstream:
    R2 ToySceneDataset provides current_points and previous_points.
Downstream:
    compare_completion will compare merged_points with complete_points.
Not responsible for:
    pose alignment, voxelization, k, neural networks, or loss functions.
Key shapes:
    input/output point clouds are N x 3 / K x 3 arrays.
"""

# Standard library: create the fixed evaluation-result box and describe inputs.
from dataclasses import dataclass
from typing import Iterable

# NumPy: normalize, stack, and deduplicate point rows.
import numpy as np


@dataclass(frozen=True)
class CompletionEvaluation:
    """Evaluation result for one estimated point cloud against complete ground truth."""

    missing_points: np.ndarray
    extra_points: np.ndarray
    coverage: float
    precision: float


def _as_point_array(points: Iterable[Iterable[float]], name: str) -> np.ndarray:
    """Normalize an input and enforce the raw point-cloud contract N x 3."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    return arr


def merge_point_clouds(
    current_points: Iterable[Iterable[float]],
    previous_points: Iterable[Iterable[float]],
) -> np.ndarray:
    """Merge current/history [x, y, z] rows and remove duplicate points."""
    current = _as_point_array(current_points, "current_points")
    previous = _as_point_array(previous_points, "previous_points")
    combined = np.vstack([current, previous])
    return np.unique(combined, axis=0)


def _point_array_from_set(points: set[tuple[float, float, float]]) -> np.ndarray:
    """Convert exact toy point tuples back to an N x 3 point-cloud array."""
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(sorted(points), dtype=float)


def compare_completion(
    merged_points: Iterable[Iterable[float]],
    complete_points: Iterable[Iterable[float]],
) -> CompletionEvaluation:
    """Compare exact toy point clouds and report missing/extra points and ratios."""
    merged = _as_point_array(merged_points, "merged_points")
    complete = _as_point_array(complete_points, "complete_points")

    merged_set = {tuple(point) for point in merged}
    complete_set = {tuple(point) for point in complete}
    correct_points = merged_set & complete_set
    missing_points = complete_set - merged_set
    extra_points = merged_set - complete_set

    return CompletionEvaluation(
        missing_points=_point_array_from_set(missing_points),
        extra_points=_point_array_from_set(extra_points),
        coverage=len(correct_points) / len(complete_set),
        precision=len(correct_points) / len(merged_set),
    )
