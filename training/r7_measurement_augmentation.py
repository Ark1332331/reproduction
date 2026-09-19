"""Paper Section III-E measurement-side augmentation primitives."""

from __future__ import annotations

import numpy as np


def augment_measurement_points(
    points: np.ndarray,
    rng: np.random.Generator,
    position_noise: float = 0.05,
    tilt_degrees: float = 1.0,
    height_patch_noise: float = 0.0,
    height_patch_count: int = 0,
    drop_patch_count: int = 0,
    patch_size: float = 0.4,
    drop_probability: float = 0.0,
    outlier_count: int = 0,
    outlier_cluster_size: int = 1,
    pose_noise: float = 0.0,
) -> np.ndarray:
    """Apply measurement corruption while keeping the target untouched."""
    if min(height_patch_noise, patch_size, pose_noise) < 0:
        raise ValueError("noise magnitudes and patch_size must be non-negative")
    if min(height_patch_count, drop_patch_count, outlier_count, outlier_cluster_size) < 0:
        raise ValueError("augmentation counts must be non-negative")
    if not 0 <= drop_probability <= 1:
        raise ValueError("drop_probability must be in [0, 1]")

    augmented = np.asarray(points, dtype=float).copy()
    if len(augmented) == 0:
        return augmented
    augmented += rng.uniform(-position_noise, position_noise, size=augmented.shape)
    tilt = np.deg2rad(rng.uniform(-tilt_degrees, tilt_degrees, size=2))
    rx = np.array(
        [[1, 0, 0], [0, np.cos(tilt[0]), -np.sin(tilt[0])], [0, np.sin(tilt[0]), np.cos(tilt[0])]]
    )
    ry = np.array(
        [[np.cos(tilt[1]), 0, np.sin(tilt[1])], [0, 1, 0], [-np.sin(tilt[1]), 0, np.cos(tilt[1])]]
    )
    augmented = augmented @ (ry @ rx).T
    for _ in range(height_patch_count):
        center = augmented[int(rng.integers(len(augmented))), :2]
        in_patch = np.all(np.abs(augmented[:, :2] - center) <= patch_size / 2, axis=1)
        augmented[in_patch, 2] += rng.uniform(-height_patch_noise, height_patch_noise)
    for _ in range(drop_patch_count):
        if len(augmented) == 0:
            break
        center = augmented[int(rng.integers(len(augmented))), :2]
        in_patch = np.all(np.abs(augmented[:, :2] - center) <= patch_size / 2, axis=1)
        augmented = augmented[~in_patch]
    augmented = augmented[rng.random(len(augmented)) >= drop_probability]
    if outlier_count:
        extent_source = augmented if len(augmented) else np.asarray(points, dtype=float)
        low, high = extent_source.min(axis=0) - 0.1, extent_source.max(axis=0) + 0.1
        centers = rng.uniform(low, high, size=(outlier_count, 3))
        clusters = [
            center + rng.uniform(-0.05, 0.05, size=(outlier_cluster_size, 3))
            for center in centers
        ]
        augmented = np.vstack([augmented, *clusters])
    if pose_noise:
        augmented += rng.uniform(-pose_noise, pose_noise, size=(1, 3))
    return augmented
