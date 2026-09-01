"""R6: convert a completed sparse map into controller-queryable terrain heights.

The paper publishes a local reconstructed point cloud slower than the locomotion
controller runs. At controller time, the robot is therefore located relative to
the latest map frame; this module performs that coordinate conversion and returns
the fixed 1.6 m x 1.0 m terrain window used by the locomotion policy.

This is a generic data contract. It does not claim that the existing Unitree
policy accepts these values: its height-scanner observation is currently disabled.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeightMap:
    """Highest reconstructed z value and visibility for each local XY cell."""

    heights: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class ControllerTerrainWindow:
    """A fixed policy-shaped window queried at a robot pose relative to one map."""

    heights: np.ndarray
    observed: np.ndarray
    center_cell: tuple[int, int]


def decode_current_voxels(
    coordinates: np.ndarray,
    offsets: np.ndarray,
    voxel_size: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Turn `[batch, x, y, z, k=0]` plus offsets back into local `[x, y, z]` points."""
    coordinate_array = np.asarray(coordinates)
    offset_array = np.asarray(offsets, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 5:
        raise ValueError("coordinates must have shape K x 5")
    if offset_array.shape != (len(coordinate_array), 3):
        raise ValueError("offsets must have shape K x 3")
    if origin_array.shape != (3,) or voxel_size <= 0:
        raise ValueError("origin must contain 3 values and voxel_size must be positive")
    if not np.all(coordinate_array[:, 4] == 0):
        raise ValueError("controller output must contain only current-frame k=0 voxels")
    return (coordinate_array[:, 1:4].astype(float) + offset_array) * voxel_size + origin_array


def height_map_from_points(points: np.ndarray, x_cells: int, y_cells: int, cell_size: float) -> HeightMap:
    """Store the highest local z point in each horizontal map cell."""
    if x_cells <= 0 or y_cells <= 0 or cell_size <= 0:
        raise ValueError("map dimensions and cell_size must be positive")
    heights = np.full((x_cells, y_cells), np.nan)
    for x, y, z in np.asarray(points, dtype=float):
        ix, iy = int(np.floor(x / cell_size)), int(np.floor(y / cell_size))
        if 0 <= ix < x_cells and 0 <= iy < y_cells:
            if np.isnan(heights[ix, iy]) or z > heights[ix, iy]:
                heights[ix, iy] = z
    return HeightMap(heights=heights, observed=np.isfinite(heights))


def relative_position_to_map_cell(
    robot_world_xy: tuple[float, float],
    map_origin_world_xy: tuple[float, float],
    cell_size: float,
) -> tuple[int, int]:
    """Locate a controller-time robot pose in the latest map's local grid."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    relative_xy = np.asarray(robot_world_xy, dtype=float) - np.asarray(map_origin_world_xy, dtype=float)
    if relative_xy.shape != (2,):
        raise ValueError("robot_world_xy and map_origin_world_xy must each contain two values")
    return tuple(np.floor(relative_xy / cell_size).astype(int))


def query_height_rectangle(
    height_map: HeightMap,
    center_cell: tuple[int, int],
    x_cells: int,
    y_cells: int,
    unknown_height: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Query a fixed rectangular local window; out-of-map cells stay unknown."""
    if x_cells <= 0 or y_cells <= 0:
        raise ValueError("window dimensions must be positive")
    heights = np.full((x_cells, y_cells), unknown_height, dtype=float)
    visible = np.zeros((x_cells, y_cells), dtype=bool)
    start_x = center_cell[0] - x_cells // 2
    start_y = center_cell[1] - y_cells // 2
    for output_x in range(x_cells):
        for output_y in range(y_cells):
            map_x, map_y = start_x + output_x, start_y + output_y
            if 0 <= map_x < height_map.heights.shape[0] and 0 <= map_y < height_map.heights.shape[1]:
                if height_map.observed[map_x, map_y]:
                    heights[output_x, output_y] = height_map.heights[map_x, map_y]
                    visible[output_x, output_y] = True
    return heights, visible


def policy_terrain_window(
    height_map: HeightMap,
    robot_world_xy: tuple[float, float],
    map_origin_world_xy: tuple[float, float],
    cell_size: float = 0.1,
    window_size_m: tuple[float, float] = (1.6, 1.0),
    unknown_height: float = 0.0,
) -> ControllerTerrainWindow:
    """Build the paper-sized policy input around the current relative robot pose."""
    window_cells = np.asarray(window_size_m, dtype=float) / cell_size
    rounded_cells = np.rint(window_cells).astype(int)
    if not np.allclose(window_cells, rounded_cells):
        raise ValueError("window_size_m must be an integer multiple of cell_size")
    center_cell = relative_position_to_map_cell(robot_world_xy, map_origin_world_xy, cell_size)
    heights, observed = query_height_rectangle(
        height_map,
        center_cell,
        x_cells=int(rounded_cells[0]),
        y_cells=int(rounded_cells[1]),
        unknown_height=unknown_height,
    )
    return ControllerTerrainWindow(heights=heights, observed=observed, center_cell=center_cell)


def isaac_height_scan(
    height_map: HeightMap,
    base_xy: tuple[float, float],
    base_z: float,
    yaw: float,
    cell_size: float = 0.1,
) -> np.ndarray:
    """Sample a flat 17 x 11 = 187 vector in the ANYmal-C rough policy's height convention.

    Verified against Isaac Lab ``Isaac-Velocity-Rough-Anymal-C-Direct-v0`` (2026-08-21):
    the RayCaster grid is 1.6 m x 1.0 m at 0.1 m resolution and includes both endpoints,
    so 17 rays along x (forward, -0.8..+0.8 m) and 11 along y (left, -0.5..+0.5 m).
    Values are ``(base_z - terrain_z - 0.5).clip(-1, 1)``; flat ground reads about +0.1.
    Flattening is row-major over y (index = iy * 17 + ix) because the sensor uses
    ``torch.meshgrid(x, y, indexing="xy")`` then ``flatten()``. Unobserved cells read 0.0,
    the value the policy saw on terrain-free flat patches.

    HeightMap indexing contract: cells are absolute world coordinates -- cell (mx, my)
    covers world [mx*0.1, (mx+1)*0.1) x [my*0.1, (my+1)*0.1). The map must therefore be
    anchored so that the whole 1.6 m x 1.0 m window around the base lies inside
    non-negative indices (e.g. a local map frame with the robot at (1.6, 1.6) m, as in
    the R6 pipeline); a window that falls on negative indices silently reads 0.0.
    """
    xs = np.arange(-0.8, 0.8 + 1e-9, cell_size)
    ys = np.arange(-0.5, 0.5 + 1e-9, cell_size)
    if len(xs) != 17 or len(ys) != 11:
        raise ValueError("height scan requires 0.1 m cells over 1.6 m x 1.0 m")
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    out = np.zeros(187, dtype=float)
    for iy, dy in enumerate(ys):
        for ix, dx in enumerate(xs):
            # rotate the grid point into the world/map frame (Isaac: x forward, y left)
            wx = base_xy[0] + cos_yaw * dx - sin_yaw * dy
            wy = base_xy[1] + sin_yaw * dx + cos_yaw * dy
            mx, my = int(np.floor(wx / cell_size)), int(np.floor(wy / cell_size))
            if 0 <= mx < height_map.heights.shape[0] and 0 <= my < height_map.heights.shape[1]:
                if height_map.observed[mx, my]:
                    out[iy * 17 + ix] = np.clip(base_z - height_map.heights[mx, my] - 0.5, -1.0, 1.0)
    return out
