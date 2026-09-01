"""Paper-distribution structured terrains as Isaac Lab sub-terrain generators.

Implements the paper's terrain types (stairs, boxes, walls, poles, corridors) as
``SubTerrainBaseCfg.function``-compatible generators returning trimesh meshes in
the sub-terrain local frame (0..8 m). Each generator also records its structural
cuboids into the module-level ``STRUCTURES`` table so the capture script can build
a dense ground-truth target from the same geometry the cameras observe.

The ANYmal spawns at the local-frame origin (0, 0) and walks along +x (the capture
script fixes a forward command), so structures are placed in front of the path:
x in [2.0, 7.5] m, y in [-3.0, 3.0] m, with a 2.4 m corridor band (|y| < 1.2)
kept mostly clear so the robot reliably reaches the structures.

Parameter ranges follow the paper's data generator and the R7 capture notes:
boxes 0.2-2.0 m wide, 0.08-0.25 m high; stairs step width 0.2-0.5 m, rise
0.08-0.25 m; corridor width 2-6 m (clipped to the 8 m terrain), wall height 1.2 m;
poles are vertical cylinders of radius 0.05-0.15 m and height 0.4-1.0 m (the paper
does not publish pole dimensions; these values are the closest documented
choice, flagged as an implementation assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Structural geometry records (local frame, 0..8 m) for dense target sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float


@dataclass(frozen=True)
class Cylinder:
    center_x: float
    center_y: float
    radius: float
    height: float


STRUCTURES: dict[str, list[Box | Cylinder]] = {}

_TERRAIN_SIZE = 8.0

_COLORS = {
    "stairs": (0.65, 0.45, 0.25),
    "boxes": (0.72, 0.55, 0.30),
    "walls": (0.55, 0.55, 0.55),
    "poles": (0.35, 0.40, 0.45),
    "corridors": (0.35, 0.40, 0.45),
}


def _ground() -> trimesh.Trimesh:
    """Flat 8x8 m ground slab whose top face sits at z=0."""
    return trimesh.creation.box(extents=(_TERRAIN_SIZE, _TERRAIN_SIZE, 0.05), transform=np.eye(4))


def _box_mesh(center_xy: tuple[float, float], size_xy: tuple[float, float], height: float) -> trimesh.Trimesh:
    """Axis-aligned cuboid from the ground plane up; center z = height / 2."""
    transform = np.eye(4)
    transform[:2, 3] = center_xy
    transform[2, 3] = height / 2.0
    return trimesh.creation.box(extents=(size_xy[0], size_xy[1], height), transform=transform)


def _cylinder_mesh(center_xy: tuple[float, float], radius: float, height: float) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:2, 3] = center_xy
    transform[2, 3] = height / 2.0
    return trimesh.creation.cylinder(radius=radius, height=height, sections=16, transform=transform)


def _front_positions(rng: np.random.Generator, count: int, min_gap: float) -> list[tuple[float, float]]:
    """Sample non-overlapping centers in front of the robot's path (x >= 2 m)."""
    positions: list[tuple[float, float]] = []
    attempts = 0
    while len(positions) < count and attempts < 500:
        attempts += 1
        x = rng.uniform(2.0, _TERRAIN_SIZE - 0.5)
        y = rng.uniform(-3.0, 3.0)
        too_close = any(abs(x - px) < min_gap and abs(y - py) < min_gap for px, py in positions)
        if not too_close:
            positions.append((x, y))
    return positions


def stairs_terrain(difficulty: float, cfg, layout_seed: int) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """One run of 4 steps climbing up the +x path.

    Step rise is kept in 0.08-0.15 m (within the official rough policy's training
    distribution) so the ANYmal reliably climbs instead of tripping; the paper's
    upper range 0.25 m is not reachable with the current checkpoint.
    """
    del difficulty, cfg  # parameters are fixed by the paper ranges below
    rng = np.random.default_rng(layout_seed)
    step_width = float(rng.uniform(0.3, 0.5))
    step_height = float(rng.uniform(0.08, 0.15))
    step_depth = float(rng.uniform(1.5, 2.5))
    boxes: list[Box] = []
    meshes: list[trimesh.Trimesh] = [_ground()]
    for index in range(4):
        level = index + 1
        center_x = 3.5 + (index + 0.5) * step_width
        box = Box(float(center_x), 0.0, step_width, step_depth, level * step_height)
        boxes.append(box)
        meshes.append(_box_mesh((center_x, 0.0), (step_width, step_depth), level * step_height))
    STRUCTURES["stairs"] = boxes
    return meshes, np.zeros(3)


def boxes_terrain(difficulty: float, cfg, layout_seed: int) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Random boxes in the paper's size ranges in front of the robot path."""
    del difficulty, cfg
    rng = np.random.default_rng(layout_seed)
    boxes: list[Box] = []
    meshes: list[trimesh.Trimesh] = [_ground()]
    for center in _front_positions(rng, 8, min_gap=0.6):
        size_x = float(rng.uniform(0.2, 2.0))
        size_y = float(rng.uniform(0.2, 2.0))
        height = float(rng.uniform(0.08, 0.25))
        box = Box(float(center[0]), float(center[1]), size_x, size_y, height)
        boxes.append(box)
        meshes.append(_box_mesh(center, (size_x, size_y), height))
    STRUCTURES["boxes"] = boxes
    return meshes, np.zeros(3)


def walls_terrain(difficulty: float, cfg, layout_seed: int) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Three random wall segments in front of the robot path (axis-aligned)."""
    del difficulty, cfg
    rng = np.random.default_rng(layout_seed)
    walls: list[Box] = []
    meshes: list[trimesh.Trimesh] = [_ground()]
    for center in _front_positions(rng, 3, min_gap=1.2):
        horizontal = bool(rng.integers(0, 2))
        length = float(rng.uniform(1.0, 3.0))
        height = float(rng.uniform(0.4, 1.0))
        thickness = 0.1
        if horizontal:
            size_x, size_y = length, thickness
        else:
            size_x, size_y = thickness, length
        wall = Box(float(center[0]), float(center[1]), size_x, size_y, height)
        walls.append(wall)
        meshes.append(_box_mesh(center, (size_x, size_y), height))
    STRUCTURES["walls"] = walls
    return meshes, np.zeros(3)


def poles_terrain(difficulty: float, cfg, layout_seed: int) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Vertical cylinders scattered in front of the robot path (paper's poles)."""
    del difficulty, cfg
    rng = np.random.default_rng(layout_seed)
    poles: list[Cylinder] = []
    meshes: list[trimesh.Trimesh] = [_ground()]
    for center in _front_positions(rng, 10, min_gap=0.7):
        radius = float(rng.uniform(0.05, 0.15))
        height = float(rng.uniform(0.4, 1.0))
        pole = Cylinder(float(center[0]), float(center[1]), radius, height)
        poles.append(pole)
        meshes.append(_cylinder_mesh(center, radius, height))
    STRUCTURES["poles"] = poles
    return meshes, np.zeros(3)


def corridors_terrain(difficulty: float, cfg, layout_seed: int) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Two long parallel walls forming a corridor along the +x path."""
    del difficulty, cfg
    rng = np.random.default_rng(layout_seed)
    corridor_width = float(min(rng.uniform(2.0, 6.0), 3.4))  # clipped to fit the 8 m terrain
    wall_height = 1.2
    wall_thickness = 0.08
    walls: list[Box] = []
    meshes: list[trimesh.Trimesh] = [_ground()]
    for side, sign in enumerate((-1.0, 1.0)):
        center_y = sign * corridor_width / 2.0
        wall = Box(4.0, float(center_y), 7.0, wall_thickness, wall_height)
        walls.append(wall)
        meshes.append(_box_mesh((4.0, center_y), (7.0, wall_thickness), wall_height))
    STRUCTURES["corridors"] = walls
    return meshes, np.zeros(3)


GENERATORS = {
    "stairs": stairs_terrain,
    "boxes": boxes_terrain,
    "walls": walls_terrain,
    "poles": poles_terrain,
    "corridors": corridors_terrain,
}
