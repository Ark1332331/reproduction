"""Capture one 12-step robot-centric depth trajectory for R7 training.

Each step contains a four-camera depth measurement and a dense target sampled
from known simulated ground, box, stair, and wall geometry. Both are expressed
in a 3.2 m cube centred on a moving, turning virtual robot. The previous
estimate is not stored: R7 must produce it from the network output during
rollout.

This is a small Isaac Lab substitute for the paper's IsaacGym/ANYmal data
generator.  It provides the required temporal data contract, not the original
200000-step terrain distribution.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture a small temporal depth trajectory for R7.")
parser.add_argument("--output", required=True)
parser.add_argument("--scene-seed", type=int, default=0)
parser.add_argument("--trajectory-steps", type=int, default=12)
parser.add_argument("--settle-steps", type=int, default=4)
parser.add_argument("--points-per-camera", type=int, default=1500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth


MAP_SIZE = 3.2
MAP_CENTER = MAP_SIZE / 2


@dataclass(frozen=True)
class Box:
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float


def _add_box(boxes: list[Box], box: Box, colour: tuple[float, float, float], name: str) -> None:
    """Add one axis-aligned rectangular surface to both Isaac Lab and target metadata."""
    boxes.append(box)
    cfg = sim_utils.CuboidCfg(
        size=(box.size_x, box.size_y, box.height),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
    )
    cfg.func(name, cfg, translation=(box.center_x, box.center_y, box.height / 2))


def _spawn_scene(seed: int) -> tuple[Camera, list[Box]]:
    rng = np.random.default_rng(seed)
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DistantLightCfg(intensity=3000.0)
    light.func("/World/Light", light)
    sim_utils.create_prim("/World/Obstacles", "Xform")
    boxes: list[Box] = []
    # Paper distribution: boxes have random width/length in [0.2, 2.0] m and
    # height in [0.08, 0.25] m. We keep their geometry separately so the dense
    # target is sampled from the same scene that the cameras observe.
    for index in range(4):
        size_x, size_y = rng.uniform(.2, 2.0, size=2)
        height = float(rng.uniform(.08, .25))
        center_x, center_y = rng.uniform(.45, 2.75, size=2)
        box = Box(float(center_x), float(center_y), float(size_x), float(size_y), height)
        _add_box(boxes, box, tuple(rng.uniform(.2, .9, size=3)), f"/World/Obstacles/Box_{index:02d}")

    # Paper distribution: stairs have width [0.2, 0.5] m and rise [0.08, 0.25]
    # m. Each step is a cuboid, so both its top and vertical faces enter target.
    step_width = float(rng.uniform(.2, .5))
    step_height = float(rng.uniform(.08, .25))
    stair_origin = rng.uniform(.45, 1.35, size=2)
    for index in range(4):
        level = index + 1
        _add_box(
            boxes,
            Box(
                center_x=float(stair_origin[0] + (index + .5) * step_width),
                center_y=float(stair_origin[1]),
                size_x=step_width,
                size_y=.9,
                height=level * step_height,
            ),
            (0.65, 0.45, 0.25),
            f"/World/Obstacles/Stair_{index:02d}",
        )

    # Two long, thin walls make a corridor. The paper samples corridor widths
    # from [2, 6] m; this local capture clips the value at 3.0 m to fit its map.
    corridor_width = float(min(rng.uniform(2.0, 6.0), 3.0))
    wall_height = 1.2
    for side, sign in enumerate((-1.0, 1.0)):
        _add_box(
            boxes,
            Box(1.6, float(1.6 + sign * corridor_width / 2), 3.2, .08, wall_height),
            (0.35, 0.4, 0.45),
            f"/World/Obstacles/Wall_{side:02d}",
        )
    sim_utils.create_prim("/World/Cameras", "Xform")
    for index in range(4):
        sim_utils.create_prim(f"/World/Cameras/Camera_{index:02d}", "Xform")
    camera = Camera(
        CameraCfg(
            prim_path="/World/Cameras/Camera_.*/DepthSensor",
            update_period=0.0,
            update_latest_camera_pose=True,
            height=120,
            width=160,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(.1, 20.0),
            ),
        )
    )
    return camera, boxes


def _camera_poses(center_xy: np.ndarray, yaw: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Place front/back/left/right cameras around a moving robot center."""
    offsets = np.array(((.35, 0.0), (-.35, 0.0), (0.0, .35), (0.0, -.35)), dtype=float)
    directions = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)), dtype=float)
    rotation = np.array(((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))))
    eyes = np.column_stack((center_xy + offsets @ rotation.T, np.full(4, .9)))
    targets = np.column_stack((center_xy + 1.4 * directions @ rotation.T, np.full(4, .1)))
    return (
        torch.tensor(eyes, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _localize(points_world: np.ndarray, center_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Express world points in a robot-centred cube whose stored origin is (0,0,0)."""
    points = np.asarray(points_world, dtype=np.float32).copy()
    inverse_yaw = np.array(((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw))), dtype=np.float32)
    points[:, :2] = (points[:, :2] - center_xy) @ inverse_yaw.T
    points[:, :2] += MAP_CENTER
    in_map = np.all((points >= 0.0) & (points < MAP_SIZE), axis=1)
    return points[in_map]


def _depth_measurement(camera: Camera, center_xy: np.ndarray, yaw: float, device: str, cap: int) -> np.ndarray:
    depth_images = camera.data.output["distance_to_image_plane"]
    camera_clouds: list[np.ndarray] = []
    for index in range(len(depth_images)):
        cloud = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[index],
            depth=depth_images[index, ..., 0],
            keep_invalid=True,
            position=camera.data.pos_w[index],
            orientation=camera.data.quat_w_ros[index],
            device=device,
        )
        points = cloud.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        local = _localize(points[np.isfinite(points).all(axis=1)], center_xy, yaw)
        if len(local) == 0:
            raise RuntimeError(f"camera {index} has no finite points in the local map")
        if len(local) > cap:
            local = local[np.linspace(0, len(local) - 1, cap, dtype=int)]
        camera_clouds.append(local)
    return np.ascontiguousarray(np.concatenate(camera_clouds, axis=0), dtype=np.float32)


def _dense_target(boxes: list[Box], center_xy: np.ndarray, yaw: float, spacing: float = .05) -> np.ndarray:
    """Sample ground and all known cuboid surfaces from the simulator geometry."""
    x_values = np.arange(center_xy[0] - MAP_CENTER, center_xy[0] + MAP_CENTER, spacing)
    y_values = np.arange(center_xy[1] - MAP_CENTER, center_xy[1] + MAP_CENTER, spacing)
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    world_points = [np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))]
    for box in boxes:
        x0, x1 = box.center_x - box.size_x / 2, box.center_x + box.size_x / 2
        y0, y1 = box.center_y - box.size_y / 2, box.center_y + box.size_y / 2
        bx = np.arange(x0, x1 + spacing / 2, spacing)
        by = np.arange(y0, y1 + spacing / 2, spacing)
        top_x, top_y = np.meshgrid(bx, by, indexing="ij")
        world_points.append(np.column_stack((top_x.ravel(), top_y.ravel(), np.full(top_x.size, box.height))))
        z = np.arange(0.0, box.height + spacing / 2, spacing)
        for fixed_x in (x0, x1):
            side_y, side_z = np.meshgrid(by, z, indexing="ij")
            world_points.append(np.column_stack((np.full(side_y.size, fixed_x), side_y.ravel(), side_z.ravel())))
        for fixed_y in (y0, y1):
            side_x, side_z = np.meshgrid(bx, z, indexing="ij")
            world_points.append(np.column_stack((side_x.ravel(), np.full(side_x.size, fixed_y), side_z.ravel())))
    return _localize(np.concatenate(world_points, axis=0), center_xy, yaw)


def main() -> None:
    if args_cli.trajectory_steps <= 1 or args_cli.settle_steps <= 0 or args_cli.points_per_camera <= 0:
        raise ValueError("trajectory-steps must exceed one; settle-steps and points-per-camera must be positive")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=args_cli.device))
    camera, boxes = _spawn_scene(args_cli.scene_seed)
    sim.reset()
    centers = np.column_stack(
        (
            np.linspace(.7, 2.5, args_cli.trajectory_steps),
            np.full(args_cli.trajectory_steps, 1.6),
        )
    )
    # A smoothly turning base makes the saved yaw contract exercise the same
    # pose-alignment path that a moving ANYmal trajectory needs.
    yaws = np.linspace(-.35, .35, args_cli.trajectory_steps)
    payload: dict[str, np.ndarray | str] = {}
    previous_center = centers[0]
    previous_yaw = float(yaws[0])
    for index, (center, yaw) in enumerate(zip(centers, yaws)):
        eyes, targets = _camera_poses(center, float(yaw), sim.device)
        camera.set_world_poses_from_view(eyes, targets)
        for _ in range(args_cli.settle_steps):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
        measurement = _depth_measurement(camera, center, float(yaw), sim.device, args_cli.points_per_camera)
        target = _dense_target(boxes, center, float(yaw))
        world_translation = center - previous_center if index else np.zeros(2)
        previous_inverse_yaw = np.array(
            ((np.cos(previous_yaw), np.sin(previous_yaw)), (-np.sin(previous_yaw), np.cos(previous_yaw)))
        )
        translation = previous_inverse_yaw @ world_translation
        payload[f"measurement_{index:02d}"] = measurement
        payload[f"target_{index:02d}"] = target
        payload[f"translation_{index:02d}"] = np.array((translation[0], translation[1], 0.0), dtype=np.float32)
        payload[f"yaw_{index:02d}"] = np.array(float(yaw - previous_yaw) if index else 0.0, dtype=np.float32)
        print(f"[trajectory] step={index:02d} measurement={len(measurement)} target={len(target)}", flush=True)
        previous_center = center
        previous_yaw = float(yaw)
    payload["metadata_json"] = json.dumps(
        {
            "coordinate_frame": "robot_centric_local_map",
            "map_size_m": MAP_SIZE,
            "trajectory_steps": args_cli.trajectory_steps,
            "scene_seed": args_cli.scene_seed,
            "camera_count": 4,
            "points_per_camera": args_cli.points_per_camera,
            "ground_truth": "dense samples from known ground and cuboid simulator geometry",
            "terrain_types": ["ground", "boxes", "stairs", "corridor_walls"],
            "base_yaw": "changes smoothly; yaw_XX is current minus previous yaw",
        }
    )
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
