"""R3-b5: evaluate binary occupancy-grid completion results."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridCompletionEvaluation:
    """Counts and ratios for one batch of predicted occupancy grids."""

    missing_count: int
    extra_count: int
    coverage: float
    precision: float


def evaluate_occupancy_grid(
    estimate: np.ndarray,
    target: np.ndarray,
) -> GridCompletionEvaluation:
    """Compare binary occupancy estimates with the complete target grids."""
    if estimate.shape != target.shape:
        raise ValueError("estimate and target must have the same shape")

    estimated_occupied = estimate.astype(bool)
    target_occupied = target.astype(bool)
    true_positive = np.logical_and(estimated_occupied, target_occupied).sum()
    missing_count = np.logical_and(target_occupied, ~estimated_occupied).sum()
    extra_count = np.logical_and(estimated_occupied, ~target_occupied).sum()
    target_count = target_occupied.sum()
    estimate_count = estimated_occupied.sum()

    return GridCompletionEvaluation(
        missing_count=int(missing_count),
        extra_count=int(extra_count),
        coverage=float(true_positive / target_count) if target_count else 1.0,
        precision=float(true_positive / estimate_count) if estimate_count else 1.0,
    )
