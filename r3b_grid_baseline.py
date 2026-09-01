"""R3-b2: non-learning occupancy-grid merge baseline."""

import numpy as np


def merge_occupancy_grids(
    current_grid: np.ndarray,
    previous_grid: np.ndarray,
) -> np.ndarray:
    """Mark a cell occupied when either observation marks it occupied."""
    return np.maximum(current_grid, previous_grid)
