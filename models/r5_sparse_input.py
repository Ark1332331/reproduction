"""R5: make the paper's R1 voxel data usable by a 4D MinkowskiEngine network.

One R1 result contains:
    coords: [x_cell, y_cell, z_cell, k], shape K x 4
    features: [x_offset, y_offset, z_offset], shape K x 3

This file adds only the framework batch column:
    [batch, x_cell, y_cell, z_cell, k], shape K_total x 5

`batch` distinguishes different training samples processed together. It is an
engineering wrapper required by MinkowskiEngine, not an extra paper feature.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from representation.r1_data_representation import VoxelRepresentation


@dataclass(frozen=True)
class SparseVoxelBatch:
    """Ready-to-convert sparse data, before it enters MinkowskiEngine.

    coordinates: int32 K_total x 5, ordered [batch, x, y, z, k].
    features: float32 K_total x 3, the unchanged R1 voxel offsets.
    """

    coordinates: np.ndarray
    features: np.ndarray


def batch_voxel_representations(
    representations: Sequence[VoxelRepresentation],
) -> SparseVoxelBatch:
    """Combine R1 results into one sparse-network batch without changing R1 data.

    Each item in `representations` is one training example. Its first added
    coordinate is 0, 1, 2, ... so identical 4D voxel locations from different
    examples remain different locations to MinkowskiEngine.
    """
    if not representations:
        raise ValueError("representations must contain at least one R1 result")

    coordinate_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for batch_index, representation in enumerate(representations):
        coords, features = _validate_r1_result(representation, batch_index)
        batch_column = np.full((len(coords), 1), batch_index, dtype=np.int32)
        coordinate_parts.append(np.column_stack([batch_column, coords]).astype(np.int32))
        feature_parts.append(features.astype(np.float32))

    coordinates = np.vstack(coordinate_parts)
    features = np.vstack(feature_parts)
    _ensure_unique_coordinates(coordinates)
    return SparseVoxelBatch(coordinates=coordinates, features=features)


def make_sparse_tensor(sparse_batch: SparseVoxelBatch, device: str | None = None):
    """Convert a SparseVoxelBatch into MinkowskiEngine's real 4D SparseTensor.

    This is the exact hand-off into the paper-style sparse network. The output
    stores coordinates in `.C` and per-voxel features in `.F`.
    """
    import torch
    import MinkowskiEngine as ME

    coordinates = torch.from_numpy(sparse_batch.coordinates)
    features = torch.from_numpy(sparse_batch.features)
    if device is not None:
        features = features.to(device)
    return ME.SparseTensor(
        features=features,
        coordinates=coordinates,
        device=device,
    )


def _validate_r1_result(
    representation: VoxelRepresentation,
    batch_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the R1 output contract before it becomes framework input."""
    coords = np.asarray(representation.coords)
    features = np.asarray(representation.features)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"representation {batch_index} coords must have shape K x 4")
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f"representation {batch_index} features must have shape K x 3")
    if len(coords) != len(features):
        raise ValueError(f"representation {batch_index} coords/features must have equal length")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"representation {batch_index} coords must use integer voxel indices")
    return coords, features


def _ensure_unique_coordinates(coordinates: np.ndarray) -> None:
    """Reject ambiguous duplicate locations rather than silently merging them."""
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("sparse coordinates must be unique within each batch item")
