"""R2: construct a complete toy terrain and partial observations.

Purpose:
    Provide controlled complete/current/previous point clouds for tests.
Upstream:
    No real sensor; this file creates a small artificial step terrain.
Downstream:
    R1 voxelization and R3 completion baselines.
Key data:
    ToySceneDataset.complete_points, current_points, previous_points,
    each containing [x, y, z] rows.
"""

# Standard library: data box and type hints used by this module.
from dataclasses import dataclass
from typing import Iterable

# NumPy: store and filter the toy point clouds.
import numpy as np


@dataclass(frozen=True)
class ToySceneDataset:
    """Data box holding the complete terrain and two partial observations."""

    complete_points: np.ndarray
    current_points: np.ndarray
    previous_points: np.ndarray


def generate_step_terrain(
    x_values: Iterable[float],
    y_values: Iterable[float],
    step_x: float = 1.0,
    low_z: float = 0.0,
    high_z: float = 0.3,
) -> np.ndarray:
    """Generate the complete toy point cloud: low ground plus one step."""
    points: list[list[float]] = []
    for x in x_values:
        z = low_z if x < step_x else high_z
        for y in y_values:
            points.append([float(x), float(y), float(z)])
    return np.array(points, dtype=float)


def make_partial_observation(
    points: np.ndarray,
    x_min: float | None = None,
    x_max: float | None = None,
) -> np.ndarray:
    """Keep the points whose x coordinate lies in the requested range."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")

    mask = np.ones(len(arr), dtype=bool)
    if x_min is not None:
        mask &= arr[:, 0] >= x_min
    if x_max is not None:
        mask &= arr[:, 0] <= x_max
    return arr[mask]


def make_step_terrain_dataset(
    x_values: Iterable[float],
    y_values: Iterable[float],
    current_x_max: float,
    previous_x_min: float,
    previous_x_max: float,
    step_x: float = 1.0,
    low_z: float = 0.0,
    high_z: float = 0.3,
) -> ToySceneDataset:
    """Build one complete terrain and simulate current/history observations."""
    complete_points = generate_step_terrain(
        x_values=x_values,
        y_values=y_values,
        step_x=step_x,
        low_z=low_z,
        high_z=high_z,
    )
    current_points = make_partial_observation(
        complete_points,
        x_max=current_x_max,
    )
    previous_points = make_partial_observation(
        complete_points,
        x_min=previous_x_min,
        x_max=previous_x_max,
    )
    return ToySceneDataset(
        complete_points=complete_points,
        current_points=current_points,
        previous_points=previous_points,
    )
