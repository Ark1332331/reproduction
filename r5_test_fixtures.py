"""Small deterministic sparse samples used by the R5 model contract tests.

These fixtures are intentionally test-only.  They are not a training dataset
and do not belong to the paper reproduction data path.
"""

from dataclasses import dataclass

import numpy as np

from r1_data_representation import points_to_voxel_representation
from r5_sparse_input import batch_voxel_representations, make_sparse_tensor


@dataclass(frozen=True)
class StructuredTerrainSample:
    complete_points: np.ndarray
    current_points: np.ndarray
    previous_points: np.ndarray


@dataclass(frozen=True)
class PreparedSparseSample:
    input_tensor: object
    target_batch: object


def make_structured_terrain_sample() -> StructuredTerrainSample:
    """Return a tiny two-height-step sample that exercises all sparse scales."""
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1],
            [0.6, 0.1, 0.1],
            [0.6, 0.6, 0.1],
            [1.1, 0.1, 0.6],
            [1.1, 0.6, 0.6],
            [1.6, 0.1, 0.6],
            [1.6, 0.6, 0.6],
        ],
        dtype=np.float32,
    )
    current = points[points[:, 0] <= 1.1]
    previous = points[(points[:, 0] >= 0.6) & (points[:, 0] <= 1.1)]
    return StructuredTerrainSample(
        complete_points=points,
        current_points=current,
        previous_points=previous,
    )


def prepare_sparse_sample(
    sample: StructuredTerrainSample,
    *,
    voxel_size: float = 0.5,
    grid_size: int = 8,
) -> PreparedSparseSample:
    """Convert a fixture into the real R1 -> MinkowskiEngine hand-off."""
    empty_previous = np.empty((0, 3), dtype=np.float32)
    input_representation = points_to_voxel_representation(
        sample.current_points,
        sample.previous_points,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    target_representation = points_to_voxel_representation(
        sample.complete_points,
        empty_previous,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    input_batch = batch_voxel_representations([input_representation])
    target_batch = batch_voxel_representations([target_representation])
    return PreparedSparseSample(
        input_tensor=make_sparse_tensor(input_batch),
        target_batch=target_batch,
    )
