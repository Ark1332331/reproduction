"""Training-time augmentations stated in Section III-E of the NSR paper.

The IsaacLab collector stores clean simulator measurements and targets.  This
module applies the paper's measurement-side corruption only while preparing a
training trajectory; targets remain clean.  Reflection is applied consistently
to the whole trajectory, including the pose delta used to align feedback.
"""

from collections.abc import Sequence

import numpy as np

from training.r7_measurement_augmentation import augment_measurement_points
from rollout.r7_autoregressive_rollout import TemporalTerrainStep
from configs.paper_config import PAPER_MAP_SIZE_M


def augment_training_trajectory(
    trajectory: Sequence[TemporalTerrainStep],
    rng: np.random.Generator,
    *,
    mirror: bool = True,
) -> tuple[TemporalTerrainStep, ...]:
    """Return one deterministic paper-style augmented view of a trajectory."""

    mirror_x = bool(rng.integers(0, 2)) if mirror else False
    mirror_y = bool(rng.integers(0, 2)) if mirror else False
    output: list[TemporalTerrainStep] = []
    for step in trajectory:
        current = _mirror_points(step.current_measurement, mirror_x, mirror_y)
        target = _mirror_points(step.target_points, mirror_x, mirror_y)
        current = augment_measurement_points(
            current,
            rng,
            position_noise=0.05,
            tilt_degrees=1.0,
            height_patch_noise=0.05,
            height_patch_count=1,
            drop_patch_count=1,
            outlier_count=1,
            outlier_cluster_size=3,
            pose_noise=0.05,
        )
        translation = np.asarray(step.previous_to_current_translation, dtype=float)
        if mirror_x:
            translation[0] *= -1.0
        if mirror_y:
            translation[1] *= -1.0
        yaw = step.previous_to_current_yaw
        if mirror_x != mirror_y:
            yaw *= -1.0
        output.append(
            TemporalTerrainStep(
                current_measurement=current,
                target_points=target,
                previous_to_current_translation=tuple(float(value) for value in translation),
                previous_to_current_yaw=float(yaw),
            )
        )
    return tuple(output)


def _mirror_points(points: np.ndarray, mirror_x: bool, mirror_y: bool) -> np.ndarray:
    """Reflect local-map points around the robot-centred paper map centre."""

    mirrored = np.asarray(points, dtype=float).copy()
    if mirror_x:
        mirrored[:, 0] = PAPER_MAP_SIZE_M - mirrored[:, 0]
    if mirror_y:
        mirrored[:, 1] = PAPER_MAP_SIZE_M - mirrored[:, 1]
    return mirrored
