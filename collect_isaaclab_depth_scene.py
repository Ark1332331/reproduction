"""Capture one real depth-camera scene in Isaac Lab for the R5 pipeline.

Run this script from Isaac Lab, not from the MinkowskiEngine environment:

    cd /home/ark/projects/IsaacLab
    ./isaaclab.sh -p /media/ark/Data/devpy/projects/allinone/reproduction/collect_isaaclab_depth_scene.py \\
        --headless --enable_cameras --output /media/ark/Data/devpy/projects/allinone/reproduction/data/isaac_scene_000.npz

The saved arrays are world-aligned float32 Nx3 clouds:
    current_points  = one partial depth-camera view
    previous_points = a nearby earlier view, already expressed in the same world frame
    complete_points = union of four virtual depth-camera views, a simulator-only reference target

This is a sensor/data-contract milestone. It does not yet reproduce the paper's
moving robot, four on-robot camera rig, or IsaacGym terrain distribution.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture a multi-view depth point-cloud scene for R5.")
parser.add_argument("--output", required=True, help="Destination .npz file.")
parser.add_argument("--scene-seed", type=int, default=0)
parser.add_argument("--settle-steps", type=int, default=8)
parser.add_argument(
    "--points-per-camera",
    type=int,
    default=5000,
    help="Deterministic cap after depth projection; R1 voxelization removes remaining duplicates.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth


MAP_MIN_WORLD = np.array((0.0, 0.0, 0.0), dtype=np.float32)
MAP_MAX_WORLD = np.array((3.2, 3.2, 3.2), dtype=np.float32)


def _spawn_scene(seed: int) -> Camera:
    """Create a static 3.2 m ground/box/wall scene and four depth cameras."""
    rng = np.random.default_rng(seed)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DistantLightCfg(intensity=3000.0))
    sim_utils.create_prim("/World/Obstacles", "Xform")
    for index in range(5):
        size_xy = rng.uniform(0.25, 0.65, size=2)
        height = float(rng.uniform(0.15, 0.75))
        location = rng.uniform(0.35, 2.85, size=2)
        box = sim_utils.CuboidCfg(
            size=(float(size_xy[0]), float(size_xy[1]), height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rng.uniform(0.2, 0.9, size=3))),
        )
        box.func(
            f"/World/Obstacles/Box_{index:02d}",
            box,
            translation=(float(location[0]), float(location[1]), height / 2),
        )

    sim_utils.create_prim("/World/Cameras", "Xform")
    for index in range(4):
        sim_utils.create_prim(f"/World/Cameras/Camera_{index:02d}", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Cameras/Camera_.*/DepthSensor",
        update_period=0.0,
        # Point-cloud conversion needs the pose belonging to the rendered frame.
        # CameraCfg otherwise intentionally retains its initialization pose.
        update_latest_camera_pose=True,
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
    )
    return Camera(cfg=camera_cfg)


def _depth_clouds(camera: Camera, device: str) -> list[np.ndarray]:
    """Convert each Isaac Lab depth image into a world-frame Nx3 cloud."""
    depth_images = camera.data.output["distance_to_image_plane"]
    clouds: list[np.ndarray] = []
    for index in range(len(depth_images)):
        cloud = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[index],
            depth=depth_images[index, ..., 0],
            # In Isaac Lab 2.3, the helper's default invalid-point filter treats
            # a 2D depth image row as one point. Keep raw values, flatten to a
            # point list below, then apply the intended per-point finite check.
            keep_invalid=True,
            position=camera.data.pos_w[index],
            orientation=camera.data.quat_w_ros[index],
            device=device,
        )
        world_points = cloud.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        world_finite = np.isfinite(world_points).all(axis=1)
        points = world_points[world_finite]
        in_map = np.all((points >= MAP_MIN_WORLD) & (points < MAP_MAX_WORLD), axis=1)
        map_points = points[in_map]
        if len(map_points) == 0:
            raise RuntimeError(f"camera {index} produced no points inside the 3.2 m local map")
        capped = _subsample_points(map_points, args_cli.points_per_camera)
        print(
            f"[capture] camera {index}: finite={len(points)}, local-map={len(map_points)}, kept={len(capped)}",
            flush=True,
        )
        clouds.append(capped)
    return clouds


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Keep a deterministic spread of depth points before expensive map processing."""
    if max_points <= 0:
        raise ValueError("--points-per-camera must be positive")
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return points[indices]


def main() -> None:
    if args_cli.settle_steps <= 0 or args_cli.points_per_camera <= 0:
        raise ValueError("--settle-steps and --points-per-camera must be positive")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    print("[capture] simulation context ready", flush=True)
    camera = _spawn_scene(args_cli.scene_seed)
    print("[capture] scene and four camera prims created", flush=True)
    sim.reset()
    print("[capture] simulator reset", flush=True)
    eyes = torch.tensor(
        [[-1.0, 1.6, 1.5], [0.6, -1.0, 1.4], [4.2, 1.6, 1.5], [1.6, 4.2, 1.4]],
        device=sim.device,
    )
    targets = torch.tensor([[1.6, 1.6, 0.35]] * 4, device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)
    print("[capture] camera poses set", flush=True)
    for _ in range(args_cli.settle_steps):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    print(f"[capture] completed {args_cli.settle_steps} simulation steps", flush=True)

    depth_images = camera.data.output["distance_to_image_plane"]
    finite_depth = depth_images[torch.isfinite(depth_images)]
    if len(finite_depth) == 0:
        depth_summary = "no finite depth values"
    else:
        depth_summary = f"min={finite_depth.min().item():.3f}, max={finite_depth.max().item():.3f}"
    print(
        f"[capture] depth tensor shape={tuple(depth_images.shape)}, "
        f"finite={len(finite_depth)}/{depth_images.numel()}, {depth_summary}",
        flush=True,
    )
    clouds = _depth_clouds(camera, sim.device)
    print("[capture] converted depth images to point clouds", flush=True)
    # Exact float-coordinate de-duplication can dominate runtime for renderer
    # output. R1 already resolves occupied voxels, so retain the capped views.
    print("[capture] concatenating four capped clouds", flush=True)
    complete = np.ascontiguousarray(np.concatenate(clouds, axis=0), dtype=np.float32)
    print(f"[capture] complete cloud shape={complete.shape}", flush=True)
    if any(len(cloud) == 0 for cloud in clouds) or len(complete) == 0:
        raise RuntimeError("depth camera produced an empty cloud; check --enable_cameras and renderer logs")
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The capped arrays are small. Uncompressed NPZ avoids making a renderer smoke
    # test depend on CPU compression, while remaining portable to the R5 environment.
    print(f"[capture] writing {output}", flush=True)
    np.savez(
        output,
        complete_points=complete,
        current_points=clouds[0],
        previous_points=clouds[1],
        metadata_json=json.dumps(
            {
                "coordinate_frame": "world",
                "map_bounds_world": [MAP_MIN_WORLD.tolist(), MAP_MAX_WORLD.tolist()],
                "current_camera_index": 0,
                "previous_camera_index": 1,
                "complete_camera_indices": [0, 1, 2, 3],
                "scene_seed": args_cli.scene_seed,
                "points_per_camera": args_cli.points_per_camera,
            }
        ),
    )
    print(f"saved {output}: current={len(clouds[0])}, previous={len(clouds[1])}, complete={len(complete)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
