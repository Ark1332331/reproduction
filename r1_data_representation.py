"""R1: convert aligned point clouds into the paper's voxel representation.

Purpose:
    Turn current/previous [x, y, z] points into c_i, f_i, and k.
Upstream:
    Raw point clouds; previous points may need pose alignment first.
Downstream:
    Later sparse-network input or R1/R2 integration tests.
Key shapes:
    input N x 3, coords K x 4, features K x 3.
"""

# Standard library: data box and type hints used by this module.
from dataclasses import dataclass
from typing import Iterable

# NumPy: array math, shapes, masks, and voxel calculations.
import numpy as np

from paper_config import PAPER_GRID_SIZE, PAPER_VOXEL_SIZE_M


@dataclass(frozen=True)
class VoxelRepresentation:
    """Data box for voxel coordinates, within-voxel features, and drops."""

    coords: np.ndarray
    features: np.ndarray
    dropped_count: int


def align_previous_points_to_current(
    previous_points: np.ndarray,
    translation: Iterable[float],
    yaw: float,
    rotation_center: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Express previous-frame N x 3 points in the current robot frame.

    This function only performs the small R1.5 alignment step. It does not
    voxelize points or add the time label k.

    ``translation`` is the current robot displacement expressed in the
    *previous* local axes.  ``rotation_center`` is the robot location in the
    chosen map coordinates.  It defaults to the coordinate origin, preserving
    the simple point-cloud contract used by R1.  A 0..3.2 m robot-centred map,
    however, stores the robot at (1.6, 1.6, 1.6), so yaw must rotate about that
    point rather than about the map's bottom-left corner.
    """
    points = _as_points(previous_points, "previous_points")
    translation_arr = np.asarray(tuple(translation), dtype=float)
    rotation_center_arr = np.asarray(tuple(rotation_center), dtype=float)
    if translation_arr.shape != (3,):
        raise ValueError("translation must contain exactly 3 values")
    if rotation_center_arr.shape != (3,):
        raise ValueError("rotation_center must contain exactly 3 values")

    shifted = points - rotation_center_arr - translation_arr
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    inverse_yaw_rotation = np.array(
        [
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw],
        ],
        dtype=float,
    )

    aligned = shifted.copy()
    aligned[:, :2] = shifted[:, :2] @ inverse_yaw_rotation.T
    aligned += rotation_center_arr
    return aligned


def points_to_voxel_representation_with_aligned_previous(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    translation: Iterable[float],
    yaw: float,
    voxel_size: float = PAPER_VOXEL_SIZE_M,
    grid_size: int = PAPER_GRID_SIZE,
    origin: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_center: Iterable[float] = (0.0, 0.0, 0.0),
) -> VoxelRepresentation:
    """Run the pipeline: align previous points, then voxelize both clouds."""
    aligned_previous = align_previous_points_to_current(
        previous_points=previous_points,
        translation=translation,
        yaw=yaw,
        rotation_center=rotation_center,
    )
    return points_to_voxel_representation(
        current_points=current_points,
        previous_points=aligned_previous,
        voxel_size=voxel_size,
        grid_size=grid_size,
        origin=origin,
    )


def points_to_voxel_representation(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    voxel_size: float = PAPER_VOXEL_SIZE_M,
    grid_size: int = PAPER_GRID_SIZE,
    origin: Iterable[float] = (0.0, 0.0, 0.0),
) -> VoxelRepresentation:
    """Convert current/previous N x 3 clouds into c_i, f_i, and k.

    Output coords are K x 4: [x_cell, y_cell, z_cell, k].
    Output features are K x 3: each point's offset inside its voxel.
    """
    current = _as_points(current_points, "current_points")
    previous = _as_points(previous_points, "previous_points")
    origin_arr = np.asarray(tuple(origin), dtype=float)
    if origin_arr.shape != (3,):
        raise ValueError("origin must contain exactly 3 values")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    current_rep = _convert_one_cloud(current, 0, voxel_size, grid_size, origin_arr)
    previous_rep = _convert_one_cloud(previous, 1, voxel_size, grid_size, origin_arr)

    coords = np.vstack([current_rep.coords, previous_rep.coords])
    features = np.vstack([current_rep.features, previous_rep.features])
    dropped_count = current_rep.dropped_count + previous_rep.dropped_count
    return VoxelRepresentation(coords=coords, features=features, dropped_count=dropped_count)


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    """Normalize an input and enforce the point-cloud contract N x 3."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    return arr


def _convert_one_cloud(
    points: np.ndarray,
    time_index: int,
    voxel_size: float,
    grid_size: int,
    origin: np.ndarray,
) -> VoxelRepresentation:
    """Convert one cloud, using time_index 0 for current or 1 for previous."""
    if len(points) == 0:
        return VoxelRepresentation(
            coords=np.empty((0, 4), dtype=int),
            features=np.empty((0, 3), dtype=float),
            dropped_count=0,
        )

    scaled = (points - origin) / voxel_size
    cells = np.floor(scaled).astype(int)
    valid_mask = np.all((cells >= 0) & (cells < grid_size), axis=1)
    valid_cells = cells[valid_mask]
    valid_scaled = scaled[valid_mask]
    dropped_count = int(np.count_nonzero(~valid_mask))

    if len(valid_cells) == 0:
        return VoxelRepresentation(
            coords=np.empty((0, 4), dtype=int),
            features=np.empty((0, 3), dtype=float),
            dropped_count=dropped_count,
        )

    unique_cells, centroid_scaled = _centroids_by_cell(valid_cells, valid_scaled)
    coords = np.column_stack(
        [unique_cells, np.full(len(unique_cells), time_index, dtype=int)]
    )
    features = centroid_scaled - unique_cells
    return VoxelRepresentation(coords=coords, features=features, dropped_count=dropped_count)


def _centroids_by_cell(cells: np.ndarray, scaled_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group points by voxel and return each voxel plus its centroid.

    ``np.unique`` keeps this hot path in compiled NumPy code instead of
    constructing one Python dictionary entry per point.  The explicit reorder
    preserves the previous first-occurrence ordering, so downstream sparse
    coordinate ordering remains deterministic and backwards compatible.
    """
    unique_cells, first_indices, inverse = np.unique(
        cells, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_indices)
    sums = np.zeros((len(unique_cells), 3), dtype=float)
    np.add.at(sums, inverse, scaled_points)
    counts = np.bincount(inverse, minlength=len(unique_cells)).astype(float)
    centroids = (sums / counts[:, None])[order]
    return unique_cells[order], centroids


def demo() -> VoxelRepresentation:
    """Build a tiny hand-checkable example for direct script execution."""
    current_points = np.array(
        [
            [0.12, 0.10, 0.10],
            [0.80, 0.50, 0.20],
        ],
        dtype=float,
    )
    previous_points = np.array(
        [
            [0.00, 0.05, 0.10],
            [3.25, 0.10, 0.10],
        ],
        dtype=float,
    )
    return points_to_voxel_representation(current_points, previous_points)


if __name__ == "__main__":
    result = demo()
    print("coords (c_i = [x_cell, y_cell, z_cell, k])")
    print(result.coords)
    print()
    print("features (f_i = [x_offset, y_offset, z_offset])")
    print(result.features)
    print()
    print(f"dropped_count = {result.dropped_count}")
