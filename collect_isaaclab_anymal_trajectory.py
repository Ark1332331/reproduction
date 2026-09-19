"""Capture paper-distribution ANYmal depth trajectories for R7 training.

The robot is a real physics ANYmal-C driven by the official rough-terrain policy
checkpoint; the terrain is one of the paper's structured types (stairs, boxes,
walls, poles, corridors) generated as an Isaac Lab sub-terrain. Four depth cameras
are declared as scene sensors mounted on the base (via ``env_cfg.scene``), so the
InteractiveScene creates, clones and updates them consistently. Each captured
frame stores the robot-centric local map under the R7 contract (3.2 m cube, robot
at (1.6, 1.6), translation/yaw between frames).

The dense target is sampled from the same structural geometry the terrain
generator placed (module ``paper_terrains.STRUCTURES`` shifted by the terrain
origin), so measurement and target come from one scene.

Usage (headless, GPU visible, TERM=xterm):

    cd /home/ark/projects/IsaacLab
    conda activate isaaclab
    ./isaaclab.sh -p /media/ark/Data/devpy/projects/allinone/reproduction/collect_isaaclab_anymal_trajectory.py \
        --headless --enable_cameras --terrain stairs --seed 0 \
        --output /media/ark/Data/devpy/projects/allinone/reproduction/data/isaac_anymal_stairs_s0.npz
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher
from paper_config import (
    PAPER_CAMERA_COUNT,
    PAPER_CAMERA_TILT_DEGREES,
    PAPER_GRID_SIZE,
    PAPER_MAP_SIZE_M,
    PAPER_ROLLOUT_STEPS,
    PAPER_VOXEL_SIZE_M,
    REPRODUCTION_CAPTURE_SCHEMA_VERSION,
    REPRODUCTION_TERRAIN_PROFILE,
)

parser = argparse.ArgumentParser(description="Capture paper-distribution ANYmal depth trajectories.")
parser.add_argument("--terrain", choices=("stairs", "boxes", "walls", "poles", "corridors", "default"), required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", required=True)
parser.add_argument("--trajectory-steps", type=int, default=PAPER_ROLLOUT_STEPS)
parser.add_argument("--frames-per-step", type=int, default=25, help="Physics env steps between captured frames.")
parser.add_argument("--settle-env-steps", type=int, default=50, help="Steps to let the robot stand before capture.")
parser.add_argument("--points-per-camera", type=int, default=1500)
parser.add_argument(
    "--random-motion", action=argparse.BooleanOptionalAction, default=True,
    help="Sample velocity/yaw command and initial yaw per trajectory (paper-aligned default).",
)
parser.add_argument("--motion-seed", type=int, default=None)
parser.add_argument("--yaw-rate-min", type=float, default=-0.35)
parser.add_argument("--yaw-rate-max", type=float, default=0.35)
parser.add_argument("--forward-velocity-min", type=float, default=0.5)
parser.add_argument("--forward-velocity-max", type=float, default=1.0)
parser.add_argument("--lateral-velocity-min", type=float, default=-0.2)
parser.add_argument("--lateral-velocity-max", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import ContactSensor  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth  # noqa: E402
from isaaclab.sensors.ray_caster import RayCaster  # noqa: E402
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.direct.anymal_c.anymal_c_env import AnymalCEnv  # noqa: E402
from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCRoughEnvCfg  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, "/media/ark/Data/devpy/projects/allinone/reproduction")
import paper_terrains  # noqa: E402
from r7_capture_provenance import capture_provenance  # noqa: E402

CHECKPOINT = (
    "/home/ark/projects/IsaacLab/.pretrained_checkpoints/rsl_rl/"
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0/checkpoint.pt"
)
TASK = "Isaac-Velocity-Rough-Anymal-C-Direct-v0"
MAP_SIZE = PAPER_MAP_SIZE_M
MAP_CENTER = MAP_SIZE / 2


def _yaw_quaternion(yaw: float, device: str) -> torch.Tensor:
    """Build Isaac's (w, x, y, z) quaternion for one planar yaw."""
    half = torch.tensor(float(yaw) * 0.5, dtype=torch.float32, device=device)
    return torch.stack((torch.cos(half), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.sin(half)))

# Camera offsets w.r.t. the base frame (world convention: +X forward, +Z up).
# The ANYmal trunk reaches ~0.5 m ahead of the base, so a camera mounted on the
# base only sees its own body close up. These cameras reach 1.2 m outside the
# trunk (pole mounts) and look 70 deg down: the sight line never crosses the body
# and sees the ground 1.2-1.9 m from the base, inside the 3.2 m local window.
# Camera geometry follows the R7 capture: front/back/left/right around the base,
# eye height 0.9 m, looking at a point 1.4 m out at 0.1 m height (~30 deg down).


class AnymalCaptureEnv(AnymalCEnv):
    """ANYmal-C rough env with four capture cameras following the base.

    The cameras are plain Xform prims under ``/World/Cameras`` created inside
    ``_setup_scene`` (before the simulation starts, like the R7 capture script).
    Each frame the capture loop writes their world poses from the live base pose,
    then the next ``env.step`` renders them (world-pose writes go through fabric,
    which is safe while the simulation runs).
    """

    def _setup_scene(self):
        # replicate AnymalCEnv._setup_scene up to the clone step
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # capture cameras as standalone xforms (outside the cloned envs)
        self._capture_cameras: list[Camera] = []
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
        self.scene.sensors["capture_cameras"] = camera
        self._capture_cameras.append(camera)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _camera_poses(self, base_xy: np.ndarray, yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Place front/back/left/right cameras around the base (R7 geometry)."""
        offsets = np.array(((.35, 0.0), (-.35, 0.0), (0.0, .35), (0.0, -.35)), dtype=float)
        directions = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)), dtype=float)
        rotation = np.array(((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))))
        eyes = np.column_stack((base_xy + offsets @ rotation.T, np.full(4, 0.9)))
        targets = np.column_stack((base_xy + 1.4 * directions @ rotation.T, np.full(4, 0.1)))
        return (
            torch.tensor(eyes, dtype=torch.float32, device=self.device),
            torch.tensor(targets, dtype=torch.float32, device=self.device),
        )


def _make_terrain_cfg(terrain: str, seed: int) -> TerrainGeneratorCfg:
    # NOTE: the sub-terrain function must be a plain function, not functools.partial:
    # TerrainGenerator hashes cfg.to_dict(), whose callable_to_string() reads
    # `__name__`, which partial objects do not have (silently aborts the app).
    def _generate(difficulty: float, cfg):
        return paper_terrains.GENERATORS[terrain](difficulty, cfg, layout_seed=seed)

    _generate.__name__ = f"{terrain}_layout{seed}"
    sub = SubTerrainBaseCfg()
    sub.function = _generate
    sub.proportion = 1.0
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=10.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        seed=seed,
        sub_terrains={terrain: sub},
    )


def _localize(points_world: np.ndarray, center_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Express world points in a robot-centred cube whose stored origin is (0,0,0)."""
    points = np.asarray(points_world, dtype=np.float32).copy()
    inverse_yaw = np.array(((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw))), dtype=np.float32)
    points[:, :2] = (points[:, :2] - center_xy) @ inverse_yaw.T
    points[:, :2] += MAP_CENTER
    in_map = np.all((points >= 0.0) & (points < MAP_SIZE), axis=1)
    return points[in_map]


def _depth_measurement(
    camera: Camera, center_xy: np.ndarray, yaw: float, device: str, cap: int
) -> np.ndarray:
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


def _dense_target(
    structures: list, ground_z: float, center_xy: np.ndarray, yaw: float, spacing: float = 0.05
) -> np.ndarray:
    """Sample ground plus known box/cylinder geometry, offset by the terrain origin."""
    x_values = np.arange(center_xy[0] - MAP_CENTER, center_xy[0] + MAP_CENTER, spacing)
    y_values = np.arange(center_xy[1] - MAP_CENTER, center_xy[1] + MAP_CENTER, spacing)
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    world_points = [np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, ground_z)))]
    for item in structures:
        if isinstance(item, paper_terrains.Box):
            x0, x1 = item.center_x - item.size_x / 2, item.center_x + item.size_x / 2
            y0, y1 = item.center_y - item.size_y / 2, item.center_y + item.size_y / 2
            bx = np.arange(x0, x1 + spacing / 2, spacing)
            by = np.arange(y0, y1 + spacing / 2, spacing)
            top_x, top_y = np.meshgrid(bx, by, indexing="ij")
            world_points.append(
                np.column_stack((top_x.ravel(), top_y.ravel(), np.full(top_x.size, ground_z + item.height)))
            )
            z = np.arange(0.0, item.height + spacing / 2, spacing)
            for fixed_x in (x0, x1):
                side_y, side_z = np.meshgrid(by, z, indexing="ij")
                world_points.append(
                    np.column_stack((np.full(side_y.size, fixed_x), side_y.ravel(), side_z.ravel() + ground_z))
                )
            for fixed_y in (y0, y1):
                side_x, side_z = np.meshgrid(bx, z, indexing="ij")
                world_points.append(
                    np.column_stack((side_x.ravel(), np.full(side_x.size, fixed_y), side_z.ravel() + ground_z))
                )
        elif isinstance(item, paper_terrains.Cylinder):
            theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
            z = np.arange(0.0, item.height + spacing / 2, spacing)
            ring_x = item.center_x + item.radius * np.cos(theta)
            ring_y = item.center_y + item.radius * np.sin(theta)
            side_xy = np.column_stack((np.tile(ring_x, len(z)), np.tile(ring_y, len(z))))
            side_z = np.repeat(z, len(theta))
            world_points.append(np.column_stack((side_xy[:, 0], side_xy[:, 1], side_z + ground_z)))
            radii = np.arange(0.0, item.radius + spacing / 2, spacing)
            top_x = item.center_x + np.outer(radii, np.cos(theta)).ravel()
            top_y = item.center_y + np.outer(radii, np.sin(theta)).ravel()
            world_points.append(np.column_stack((top_x, top_y, np.full(top_x.size, ground_z + item.height))))
        else:
            raise TypeError(f"unknown structure type: {type(item)}")
    return _localize(np.concatenate(world_points, axis=0), center_xy, yaw)


def main() -> None:
    if args_cli.trajectory_steps <= 1 or args_cli.frames_per_step <= 0:
        raise ValueError("trajectory-steps must exceed one and frames-per-step must be positive")

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if args_cli.terrain != "default":
        env_cfg.terrain.terrain_generator = _make_terrain_cfg(args_cli.terrain, args_cli.seed)
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.num_envs = 1
    # long episodes so the capture never hits a time-out mid-trajectory
    env_cfg.episode_length_s = 60.0

    env = AnymalCaptureEnv(env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] checkpoint loaded (iter {runner.current_learning_iteration}); terrain={args_cli.terrain} "
          f"seed={args_cli.seed}", flush=True)

    camera = env.unwrapped._capture_cameras[0]
    robot = env.unwrapped._robot
    terrain = env.unwrapped._terrain
    env_origin = terrain.env_origins[0].detach().cpu().numpy()
    terrain_origin = terrain.terrain_origins[0, 0].detach().cpu().numpy()
    print(f"[INFO] env origin {env_origin}  terrain origin {terrain_origin}")

    motion_rng = np.random.default_rng(
        args_cli.motion_seed if args_cli.motion_seed is not None else args_cli.seed + 1_000_003
    )
    if args_cli.random_motion:
        motion_command = np.array(
            (
                motion_rng.uniform(args_cli.forward_velocity_min, args_cli.forward_velocity_max),
                motion_rng.uniform(args_cli.lateral_velocity_min, args_cli.lateral_velocity_max),
                motion_rng.uniform(args_cli.yaw_rate_min, args_cli.yaw_rate_max),
            ),
            dtype=np.float32,
        )
        initial_yaw = float(motion_rng.uniform(-np.pi / 6.0, np.pi / 6.0))
        default_root_state = robot.data.default_root_state.clone()
        default_root_state[:, :3] += env_origin
        default_root_state[:, 3:7] = _yaw_quaternion(initial_yaw, env.unwrapped.device)
        robot.write_root_pose_to_sim(default_root_state[:, :7])
    else:
        motion_command = np.array((0.8, 0.0, 0.0), dtype=np.float32)
        current_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        initial_yaw = float(
            np.arctan2(
                2.0 * (current_quat[0] * current_quat[3] + current_quat[1] * current_quat[2]),
                1.0 - 2.0 * (current_quat[2] ** 2 + current_quat[3] ** 2),
            )
        )
    env.unwrapped._commands[:] = torch.as_tensor(motion_command[None], dtype=torch.float32, device=env.unwrapped.device)

    # Let the robot settle and take its first command
    obs_td = env.get_observations()
    for _ in range(args_cli.settle_env_steps):
        with torch.inference_mode():
            actions = policy(obs_td)
            obs_td, _, _, _ = env.step(actions)
    print(f"[DBG] settle done; base pos={robot.data.root_pos_w[0].detach().cpu().numpy()}", flush=True)

    def _set_camera_pose() -> None:
        base = robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        eyes, targets = env.unwrapped._camera_poses(base[:2], yaw)
        camera.set_world_poses_from_view(eyes, targets)

    _set_camera_pose()

    payload: dict[str, np.ndarray | str] = {}
    previous_base = robot.data.root_pos_w[0].detach().cpu().numpy()
    previous_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    previous_yaw = np.arctan2(
        2.0 * (previous_quat[0] * previous_quat[3] + previous_quat[1] * previous_quat[2]),
        1.0 - 2.0 * (previous_quat[2] ** 2 + previous_quat[3] ** 2),
    )
    structures = list(paper_terrains.STRUCTURES[args_cli.terrain]) if args_cli.terrain != "default" else []
    print(f"[DBG] structures raw: {len(structures)}", flush=True)
    # structures are stored in the sub-terrain local frame; the terrain mesh lives in
    # world = local + terrain_origins (the layout transform), so shift by terrain_origin.
    if structures:
        shift = terrain_origin[:2]
        print(f"[DBG] structure shift: {shift} first={structures[0]}", flush=True)
        if isinstance(structures[0], paper_terrains.Box):
            structures = [
                paper_terrains.Box(b.center_x + shift[0], b.center_y + shift[1], b.size_x, b.size_y, b.height)
                for b in structures
            ]
        else:
            structures = [
                paper_terrains.Cylinder(c.center_x + shift[0], c.center_y + shift[1], c.radius, c.height)
                for c in structures
            ]
    ground_z = float(env_origin[2])

    captured = 0
    for frame in range(args_cli.trajectory_steps):
        terminated_this_frame = False
        for _ in range(args_cli.frames_per_step):
            with torch.inference_mode():
                actions = policy(obs_td)
                obs_td, _, dones, _ = env.step(actions)
            if bool(dones[0].item()):
                term = bool(env.unwrapped.reset_terminated[0].item())
                base_now = robot.data.root_pos_w[0].detach().cpu().numpy()
                print(f"[WARN] robot terminated at frame {frame} (died={term}) "
                      f"base=({base_now[0]:.3f},{base_now[1]:.3f},{base_now[2]:.3f})", flush=True)
                terminated_this_frame = True
                break
        base = robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        # set the pose for the next frame's rendering
        _set_camera_pose()
        if terminated_this_frame:
            break
        # the env.step loop just rendered with the pose set before the loop; read the depth
        measurement = _depth_measurement(camera, base[:2], yaw, env.unwrapped.device, args_cli.points_per_camera)
        target = _dense_target(structures, ground_z, base[:2], yaw)

        world_translation = base[:2] - previous_base[:2]
        previous_inverse_yaw = np.array(
            ((np.cos(previous_yaw), np.sin(previous_yaw)), (-np.sin(previous_yaw), np.cos(previous_yaw)))
        )
        translation = previous_inverse_yaw @ world_translation
        suffix = f"{captured:02d}"
        payload[f"measurement_{suffix}"] = measurement
        payload[f"target_{suffix}"] = target
        payload[f"translation_{suffix}"] = np.array((translation[0], translation[1], 0.0), dtype=np.float32)
        payload[f"yaw_{suffix}"] = np.array(float(yaw - previous_yaw), dtype=np.float32)
        print(f"[trajectory] frame={captured:02d} measurement={len(measurement)} target={len(target)} "
              f"base=({base[0]:.2f},{base[1]:.2f},{base[2]:.3f}) yaw={yaw:.3f}", flush=True)
        captured += 1
        previous_base = base
        previous_yaw = yaw
        if bool(dones[0].item()):
            break

    if captured < 2:
        raise RuntimeError(f"captured only {captured} frames; robot could not walk the terrain")

    payload["metadata_json"] = json.dumps({
        "capture_schema_version": REPRODUCTION_CAPTURE_SCHEMA_VERSION,
        "coordinate_frame": "robot_centric_local_map",
        "map_size_m": MAP_SIZE,
        "voxel_size_m": PAPER_VOXEL_SIZE_M,
        "grid_size": PAPER_GRID_SIZE,
        "trajectory_steps": captured,
        "scene_seed": args_cli.seed,
        "camera_count": 4,
        "camera_tilt_degrees": PAPER_CAMERA_TILT_DEGREES,
        "motion_randomized": bool(args_cli.random_motion),
        "motion_seed": int(args_cli.motion_seed if args_cli.motion_seed is not None else args_cli.seed + 1_000_003),
        "command_xyz": motion_command.tolist(),
        "initial_yaw": initial_yaw,
        "yaw_rate_range": [args_cli.yaw_rate_min, args_cli.yaw_rate_max],
        "forward_velocity_range": [args_cli.forward_velocity_min, args_cli.forward_velocity_max],
        "lateral_velocity_range": [args_cli.lateral_velocity_min, args_cli.lateral_velocity_max],
        "points_per_camera": args_cli.points_per_camera,
        "terrain": args_cli.terrain,
        "ground_truth": "dense samples from the known terrain generator geometry",
        "drive": "official Isaac-Velocity-Rough-Anymal-C-Direct-v0 checkpoint",
        "base_yaw": "yaw_XX is current minus previous yaw",
        "provenance": capture_provenance(
            collector_path=__file__,
            checkpoint_path=CHECKPOINT,
            task=TASK,
            terrain_profile=REPRODUCTION_TERRAIN_PROFILE,
            map_size_m=MAP_SIZE,
            voxel_size_m=PAPER_VOXEL_SIZE_M,
            grid_size=PAPER_GRID_SIZE,
            rollout_steps=PAPER_ROLLOUT_STEPS,
            camera_count=PAPER_CAMERA_COUNT,
            camera_tilt_degrees=PAPER_CAMERA_TILT_DEGREES,
        ),
    })
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(f"saved {output} ({captured} frames)", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        exit_code = 1
        raise
    finally:
        # In this headless capture workflow Isaac Sim may hang indefinitely in
        # ``simulation_app.close()`` after the NPZ has already been flushed.
        # A capture is an isolated subprocess, so force-exit after main() is
        # safer than leaving a completed job holding several GB of GPU memory.
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
