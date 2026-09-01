"""R5: a hand-checkable 3D terrain sample that connects R1 to sparse training.

Real data flow in this file:
    complete terrain
    -> current partial observation and previous partial observation
    -> R1 voxel inputs with k=0/k=1
    -> current-frame complete terrain target with k=0
    -> batched sparse tensors for the R5 model and loss

The points are synthetic so every coordinate can be checked by hand. Unlike
R3's 2D grid, they are real [x, y, z] point clouds and use the R1/R5 contracts.
"""

from dataclasses import dataclass

import numpy as np

from r1_data_representation import VoxelRepresentation, points_to_voxel_representation
from r5_sparse_input import SparseVoxelBatch, batch_voxel_representations, make_sparse_tensor


@dataclass(frozen=True)
class StructuredTerrainSample:
    """One full terrain plus what the current and previous scans can see."""

    complete_points: np.ndarray
    current_points: np.ndarray
    previous_points: np.ndarray


@dataclass(frozen=True)
class PreparedSparseSample:
    """The same terrain expressed at the R1, R5-input, and R5-target stages."""

    input_representation: VoxelRepresentation
    target_representation: VoxelRepresentation
    input_batch: SparseVoxelBatch
    target_batch: SparseVoxelBatch
    input_tensor: object


@dataclass(frozen=True)
class PreparedSparseDataset:
    """Many prepared samples combined into one MinkowskiEngine batch."""

    input_batch: SparseVoxelBatch
    target_batch: SparseVoxelBatch
    input_tensor: object
    sample_count: int


def make_structured_terrain_sample(
    step_height: float = 0.6,
    current_x_max: float = 1.1,
    previous_x_min: float = 0.6,
    previous_x_max: float = 1.1,
) -> StructuredTerrainSample:
    """Make a parameterized stepped terrain with a deliberately unseen far edge."""
    if step_height <= 0.1:
        raise ValueError("step_height must be above the low terrain height")
    x_values = (0.1, 0.6, 1.1, 1.6)
    y_values = (0.1, 0.6)
    complete_points = np.array(
        [
            [x, y, 0.1 if x < 1.0 else step_height]
            for x in x_values
            for y in y_values
        ],
        dtype=float,
    )
    current_points = complete_points[complete_points[:, 0] <= current_x_max]
    previous_points = complete_points[
        (complete_points[:, 0] >= previous_x_min)
        & (complete_points[:, 0] <= previous_x_max)
    ]
    return StructuredTerrainSample(
        complete_points=complete_points,
        current_points=current_points,
        previous_points=previous_points,
    )


def make_randomized_structured_samples(count: int, seed: int) -> list[StructuredTerrainSample]:
    """Generate deterministic, varied stepped-terrain observations for R5 training."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    samples=[]
    for _ in range(count):
        samples.append(make_structured_terrain_sample(
            step_height=float(rng.uniform(0.3, 0.95)),
            current_x_max=float(rng.choice((0.6, 1.1))),
            previous_x_min=float(rng.choice((0.1, 0.6))),
            previous_x_max=1.1,
        ))
    return samples


def make_randomized_urban_samples(count: int, seed: int) -> list[StructuredTerrainSample]:
    """Generate compact ground/box/wall point-cloud scenes inside the paper-like 3.2 m cube."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng=np.random.default_rng(seed); values=np.arange(0.1,3.2,0.5); samples=[]
    for _ in range(count):
        points=[[x,y,0.1] for x in values for y in values]
        bx,by=rng.choice(values[:-2]),rng.choice(values[:-2]); height=float(rng.uniform(.3,.9))
        for x in values[(values>=bx)&(values<=bx+.8)]:
            for y in values[(values>=by)&(values<=by+.8)]: points.append([x,y,height])
        wall_x=float(rng.choice(values[2:])); wall_y=float(rng.choice(values))
        for z in np.arange(.1,float(rng.uniform(.5,1.2)),.5): points.append([wall_x,wall_y,z])
        complete=np.unique(np.asarray(points,dtype=float),axis=0)
        current=complete[complete[:,0]<=float(rng.choice((1.1,1.6,2.1)))]
        previous=complete[(complete[:,0]>=.6)&(complete[:,0]<=2.6)]
        samples.append(StructuredTerrainSample(complete,current,previous))
    return samples


def load_isaaclab_depth_sample(path: str) -> StructuredTerrainSample:
    """Load one simulator capture written by ``collect_isaaclab_depth_scene.py``.

    The capture is deliberately a plain ``.npz`` contract, so Isaac Lab can run
    in its own Python environment while R5 keeps using MinkowskiEngine's one.
    All three arrays are world-aligned Nx3 point clouds. ``complete_points`` is
    a multi-view reference target; current and previous are two partial views.
    """
    with np.load(path, allow_pickle=False) as capture:
        required = ("complete_points", "current_points", "previous_points")
        missing = [name for name in required if name not in capture]
        if missing:
            raise ValueError(f"capture is missing arrays: {', '.join(missing)}")
        complete_points = _capture_points(capture["complete_points"], "complete_points")
        current_points = _capture_points(capture["current_points"], "current_points")
        previous_points = _capture_points(capture["previous_points"], "previous_points")
    return StructuredTerrainSample(complete_points, current_points, previous_points)


def _capture_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape N x 3")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def prepare_sparse_sample(
    sample: StructuredTerrainSample,
    voxel_size: float = 0.5,
    grid_size: int = 8,
) -> PreparedSparseSample:
    """Run one sample through R1 and package its input/target for R5.

    The input has current (k=0) and previous (k=1) observations. The target is
    only the complete terrain at the current time, so it contains k=0 rows only.
    """
    empty_previous = np.empty((0, 3), dtype=float)
    input_representation = points_to_voxel_representation(
        current_points=sample.current_points,
        previous_points=sample.previous_points,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    target_representation = points_to_voxel_representation(
        current_points=sample.complete_points,
        previous_points=empty_previous,
        voxel_size=voxel_size,
        grid_size=grid_size,
    )
    input_batch = batch_voxel_representations([input_representation])
    target_batch = batch_voxel_representations([target_representation])
    return PreparedSparseSample(
        input_representation=input_representation,
        target_representation=target_representation,
        input_batch=input_batch,
        target_batch=target_batch,
        input_tensor=make_sparse_tensor(input_batch),
    )


def prepare_sparse_dataset(
    samples: list[StructuredTerrainSample],
    voxel_size: float = 0.5,
    grid_size: int = 8,
    augmentation_seed: int | None = None,
    mirror: bool = True,
    device: str | None = None,
) -> PreparedSparseDataset:
    """Voxelize samples with paper-style sample and measurement augmentations.

    Mirroring changes an entire scene consistently, including ground truth.
    Sensor noise, tilt, height patches, local occlusion, outlier clusters, and
    pose disturbance affect only current/previous measurements.
    """
    if not samples:
        raise ValueError("samples must contain at least one terrain")

    input_representations: list[VoxelRepresentation] = []
    target_representations: list[VoxelRepresentation] = []
    empty_previous = np.empty((0, 3), dtype=float)
    rng = np.random.default_rng(augmentation_seed) if augmentation_seed is not None else None
    domain_size = voxel_size * grid_size
    for original_sample in samples:
        sample = original_sample
        if rng is not None and mirror:
            sample = mirror_structured_sample(
                sample,
                x_extent=domain_size,
                y_extent=domain_size,
                mirror_x=bool(rng.integers(0, 2)),
                mirror_y=bool(rng.integers(0, 2)),
            )
        current_points = sample.current_points
        previous_points = sample.previous_points
        if rng is not None:
            current_points = augment_measurement_points(
                current_points,
                rng,
                height_patch_noise=0.05,
                height_patch_count=1,
                drop_patch_count=1,
                outlier_count=1,
                outlier_cluster_size=3,
                pose_noise=0.05,
            )
            previous_points = augment_measurement_points(
                previous_points,
                rng,
                height_patch_noise=0.05,
                height_patch_count=1,
                drop_patch_count=1,
                outlier_count=1,
                outlier_cluster_size=3,
                pose_noise=0.05,
            )
        input_representations.append(
            points_to_voxel_representation(
                current_points=current_points,
                previous_points=previous_points,
                voxel_size=voxel_size,
                grid_size=grid_size,
            )
        )
        target_representations.append(
            points_to_voxel_representation(
                current_points=sample.complete_points,
                previous_points=empty_previous,
                voxel_size=voxel_size,
                grid_size=grid_size,
            )
        )

    input_batch = batch_voxel_representations(input_representations)
    target_batch = batch_voxel_representations(target_representations)
    return PreparedSparseDataset(
        input_batch=input_batch,
        target_batch=target_batch,
        input_tensor=make_sparse_tensor(input_batch, device=device),
        sample_count=len(samples),
    )


def mirror_structured_sample(
    sample: StructuredTerrainSample,
    x_extent: float,
    y_extent: float,
    mirror_x: bool,
    mirror_y: bool,
) -> StructuredTerrainSample:
    """Mirror the full current/previous/target trajectory in one shared frame."""
    if x_extent <= 0 or y_extent <= 0:
        raise ValueError("mirror extents must be positive")

    def mirror_points(points: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(points, dtype=float).copy()
        if mirror_x:
            mirrored[:, 0] = x_extent - mirrored[:, 0]
        if mirror_y:
            mirrored[:, 1] = y_extent - mirrored[:, 1]
        return mirrored

    return StructuredTerrainSample(
        complete_points=mirror_points(sample.complete_points),
        current_points=mirror_points(sample.current_points),
        previous_points=mirror_points(sample.previous_points),
    )


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
    """Apply the paper's measurement-side sensor perturbations.

    ``height_patch_count`` changes a contiguous XY patch height, while
    ``drop_patch_count`` removes contiguous XY patches. ``outlier_count`` means
    number of clusters, not number of isolated points. The clean target never
    enters this function.
    """
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
    rx = np.array([[1, 0, 0], [0, np.cos(tilt[0]), -np.sin(tilt[0])], [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    ry = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])], [0, 1, 0], [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
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
    keep = rng.random(len(augmented)) >= drop_probability
    augmented = augmented[keep]
    if outlier_count:
        # A high drop probability may remove every measured point. In that case
        # use the pre-drop extent as the outlier sampling box instead of failing.
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
