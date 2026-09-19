"""Collect several independent R7 ANYmal trajectories in one Isaac Sim process.

This is deliberately a *throughput prototype*, not a paper-scale data claim.
Unlike the older serial collector, one simulation owns ``--num-envs`` terrain
tiles.  Each tile has its own layout seed, ANYmal instance and four camera
streams.  Every surviving environment is written as a separate NPZ following
the existing R7 temporal-data contract:

    measurement_XX, target_XX, translation_XX, yaw_XX, metadata_json

The important invariant is that a camera tensor at index ``i`` belongs to robot
``i`` and target geometry from terrain tile ``i``.  The script checks those
counts at runtime before saving anything.

Example small smoke (two simultaneous robots):

    cd /home/ark/projects/IsaacLab
    conda activate isaaclab
    ./isaaclab.sh -p /media/ark/Data/devpy/projects/allinone/reproduction/collect_isaaclab_anymal_vectorized.py \\
      --headless --enable_cameras --terrain boxes --seed 20 --num-envs 2 \\
      --trajectory-steps 3 --output-dir /media/ark/Data/devpy/projects/allinone/reproduction/data/vectorized_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaaclab.app import AppLauncher
from configs.paper_config import (
    PAPER_CAMERA_COUNT,
    PAPER_CAMERA_TILT_DEGREES,
    PAPER_GRID_SIZE,
    PAPER_MAP_SIZE_M,
    PAPER_ROLLOUT_STEPS,
    PAPER_VOXEL_SIZE_M,
    REPRODUCTION_CAPTURE_SCHEMA_VERSION,
    REPRODUCTION_TERRAIN_PROFILE,
)

PARSER = argparse.ArgumentParser(description="Collect vectorized R7 ANYmal depth trajectories.")
PARSER.add_argument("--terrain", choices=("stairs", "boxes", "walls", "poles", "corridors"), required=True)
PARSER.add_argument("--seed", type=int, default=0, help="Seed of environment zero; later environments add their index.")
PARSER.add_argument("--num-envs", type=int, default=2, help="Parallel ANYmal/terrain tiles in one Isaac Sim process.")
PARSER.add_argument("--output-dir", required=True)
PARSER.add_argument("--trajectory-steps", type=int, default=PAPER_ROLLOUT_STEPS)
PARSER.add_argument("--frames-per-step", type=int, default=25)
PARSER.add_argument("--settle-env-steps", type=int, default=50)
PARSER.add_argument("--points-per-camera", type=int, default=1500)
PARSER.add_argument(
    "--random-motion", action=argparse.BooleanOptionalAction, default=True,
    help="Sample one motion command and initial yaw per trajectory (paper-aligned default).",
)
PARSER.add_argument("--motion-seed", type=int, default=None, help="Optional independent seed for --random-motion.")
PARSER.add_argument("--yaw-rate-min", type=float, default=-0.35, help="Lower yaw-rate bound for randomized motion (rad/s).")
PARSER.add_argument("--yaw-rate-max", type=float, default=0.35, help="Upper yaw-rate bound for randomized motion (rad/s).")
PARSER.add_argument("--forward-velocity-min", type=float, default=0.5)
PARSER.add_argument("--forward-velocity-max", type=float, default=1.0)
PARSER.add_argument("--lateral-velocity-min", type=float, default=-0.2)
PARSER.add_argument("--lateral-velocity-max", type=float, default=0.2)
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()

if ARGS.yaw_rate_min >= ARGS.yaw_rate_max or ARGS.forward_velocity_min >= ARGS.forward_velocity_max or ARGS.lateral_velocity_min >= ARGS.lateral_velocity_max:
    raise ValueError("motion lower bounds must be smaller than upper bounds")

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

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
from isaaclab_tasks.direct.anymal_c.anymal_c_env import AnymalCEnv  # noqa: E402
from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCRoughEnvCfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import simulation.paper_terrains as paper_terrains  # noqa: E402
from simulation.r7_capture_provenance import capture_provenance  # noqa: E402
from simulation.r7_vectorized_capture_contract import (  # noqa: E402
    CAMERA_DIRECTIONS,
    layout_seed_for_environment,
    output_path_for_environment,
    required_camera_count,
)

CHECKPOINT = (
    "/home/ark/projects/IsaacLab/.pretrained_checkpoints/rsl_rl/"
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0/checkpoint.pt"
)
TASK = "Isaac-Velocity-Rough-Anymal-C-Direct-v0"
MAP_SIZE = PAPER_MAP_SIZE_M
MAP_CENTER = MAP_SIZE / 2


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    """Extract world z-yaw from Isaac's (w, x, y, z) base quaternion."""
    return float(
        np.arctan2(
            2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
        )
    )


def _yaw_quaternions(yaws: np.ndarray, device: str) -> torch.Tensor:
    """Build Isaac's (w, x, y, z) quaternions for planar yaw offsets."""
    half = torch.as_tensor(yaws, dtype=torch.float32, device=device) * 0.5
    quaternions = torch.zeros((len(yaws), 4), dtype=torch.float32, device=device)
    quaternions[:, 0] = torch.cos(half)
    quaternions[:, 3] = torch.sin(half)
    return quaternions


def _localize(points_world: np.ndarray, center_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Express world points in the R7 robot-centred local cube [0, MAP_SIZE)."""
    points = np.asarray(points_world, dtype=np.float32).copy()
    inverse_yaw = np.array(((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw))), dtype=np.float32)
    points[:, :2] = (points[:, :2] - center_xy) @ inverse_yaw.T
    points[:, :2] += MAP_CENTER
    in_map = np.all((points >= 0.0) & (points < MAP_SIZE), axis=1)
    return points[in_map]


def _dense_target(
    structures: list, ground_z: float, center_xy: np.ndarray, yaw: float, spacing: float = 0.05
) -> np.ndarray:
    """Sample the known terrain geometry in exactly the same local frame as depth points."""
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
            world_points.append(np.column_stack((top_x.ravel(), top_y.ravel(), np.full(top_x.size, ground_z + item.height))))
            z = np.arange(0.0, item.height + spacing / 2, spacing)
            for fixed_x in (x0, x1):
                side_y, side_z = np.meshgrid(by, z, indexing="ij")
                world_points.append(np.column_stack((np.full(side_y.size, fixed_x), side_y.ravel(), side_z.ravel() + ground_z)))
            for fixed_y in (y0, y1):
                side_x, side_z = np.meshgrid(bx, z, indexing="ij")
                world_points.append(np.column_stack((side_x.ravel(), np.full(side_x.size, fixed_y), side_z.ravel() + ground_z)))
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


def _translate_structures(structures: list, shift_xy: np.ndarray) -> list:
    """Move sub-terrain-local geometry to the generated terrain tile in world coordinates."""
    translated = []
    for item in structures:
        if isinstance(item, paper_terrains.Box):
            translated.append(
                paper_terrains.Box(item.center_x + shift_xy[0], item.center_y + shift_xy[1], item.size_x, item.size_y, item.height)
            )
        elif isinstance(item, paper_terrains.Cylinder):
            translated.append(paper_terrains.Cylinder(item.center_x + shift_xy[0], item.center_y + shift_xy[1], item.radius, item.height))
        else:
            raise TypeError(f"unknown structure type: {type(item)}")
    return translated


def _make_batched_terrain_cfg(terrain: str, base_seed: int, num_envs: int) -> tuple[TerrainGeneratorCfg, list[list]]:
    """Build one tile per environment and retain the exact geometry for its target labels."""
    layouts: list[list] = []

    def _generate(difficulty: float, cfg):
        index = len(layouts)
        layout_seed = layout_seed_for_environment(base_seed, index)
        meshes, origin = paper_terrains.GENERATORS[terrain](difficulty, cfg, layout_seed=layout_seed)
        # The paper_terrains generator records the same objects used to create meshes.
        # Copy them now: the module-level scratch entry changes on the next tile.
        layouts.append(list(paper_terrains.STRUCTURES[terrain]))
        return meshes, origin

    _generate.__name__ = f"{terrain}_vectorized_seed{base_seed}"
    sub = SubTerrainBaseCfg()
    sub.function = _generate
    sub.proportion = 1.0
    return (
        TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=10.0,
            num_rows=num_envs,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            seed=base_seed,
            sub_terrains={terrain: sub},
        ),
        layouts,
    )


class VectorizedAnymalCaptureEnv(AnymalCEnv):
    """ANYmal rough terrain environment with four camera streams per cloned environment.

    The environments themselves are vectorized. This first correctness probe
    deliberately uses ``Camera`` rather than ``TiledCamera``: the latter yielded
    only infinite depth in our headless dynamic-pose setup, whereas this camera
    path is already validated by the serial collector. Rendering optimization
    is a later, separately measured step.
    """

    def _setup_scene(self):
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

        self._capture_cameras: dict[str, Camera] = {}
        # Camera creates the camera prim below this parent, but Isaac Lab
        # intentionally does not create missing parent Xforms for a regex path.
        # Create it in env_0 before clone_environments() so every clone receives
        # an identically named Capture branch.
        sim_utils.create_prim("/World/envs/env_0/Capture", "Xform")
        for direction in CAMERA_DIRECTIONS:
            camera = Camera(
                CameraCfg(
                    prim_path=f"/World/envs/env_.*/Capture/Camera_{direction}",
                    update_period=0.0,
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
            )
            self._capture_cameras[direction] = camera

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        for direction, camera in self._capture_cameras.items():
            self.scene.sensors[f"capture_{direction}"] = camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


def _set_all_camera_poses(env: VectorizedAnymalCaptureEnv) -> None:
    """Place every direction camera around its matching live robot base pose."""
    bases = env._robot.data.root_pos_w.detach().cpu().numpy()
    quaternions = env._robot.data.root_quat_w.detach().cpu().numpy()
    yaws = np.array([_yaw_from_quaternion(quaternion) for quaternion in quaternions], dtype=np.float32)
    offsets = {"front": (0.35, 0.0), "back": (-0.35, 0.0), "left": (0.0, 0.35), "right": (0.0, -0.35)}
    directions = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
    rotation = np.stack(
        (np.stack((np.cos(yaws), -np.sin(yaws)), axis=1), np.stack((np.sin(yaws), np.cos(yaws)), axis=1)), axis=1
    )
    for name, camera in env._capture_cameras.items():
        offset = np.asarray(offsets[name], dtype=np.float32)
        direction = np.asarray(directions[name], dtype=np.float32)
        eyes_xy = bases[:, :2] + np.einsum("nij,j->ni", rotation, offset)
        targets_xy = bases[:, :2] + np.einsum("nij,j->ni", rotation, 1.4 * direction)
        eyes = np.column_stack((eyes_xy, np.full(env.num_envs, 0.9, dtype=np.float32)))
        targets = np.column_stack((targets_xy, np.full(env.num_envs, 0.1, dtype=np.float32)))
        camera.set_world_poses_from_view(
            torch.as_tensor(eyes, dtype=torch.float32, device=env.device),
            torch.as_tensor(targets, dtype=torch.float32, device=env.device),
        )


def _depth_measurement(
    cameras: dict[str, Camera], environment_index: int, center_xy: np.ndarray, yaw: float, device: str, cap: int
) -> tuple[np.ndarray, int]:
    """Convert available camera observations for one environment into a local point cloud.

    A directional camera can validly contribute no points after the robot-centric
    3.2 m crop (for example when the robot is near a terrain boundary).  Treat
    that stream as a missing observation, rather than aborting every parallel
    environment.  A frame with no usable camera stream remains invalid.
    """
    clouds = []
    for direction in CAMERA_DIRECTIONS:
        camera = cameras[direction]
        depth = camera.data.output["distance_to_image_plane"][environment_index, ..., 0]
        cloud = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[environment_index],
            depth=depth,
            keep_invalid=True,
            position=camera.data.pos_w[environment_index],
            orientation=camera.data.quat_w_ros[environment_index],
            device=device,
        )
        world_points = cloud.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        local = _localize(world_points[np.isfinite(world_points).all(axis=1)], center_xy, yaw)
        if len(local) == 0:
            finite_depth = int(torch.isfinite(depth).sum().item())
            print(
                f"[MISSING_CAMERA] environment={environment_index} camera={direction} has no local map points; "
                f"finite_depth={finite_depth}/{depth.numel()} camera_pos="
                f"{camera.data.pos_w[environment_index].detach().cpu().tolist()} robot_xy={center_xy.tolist()}",
                flush=True,
            )
            continue
        if len(local) > cap:
            local = local[np.linspace(0, len(local) - 1, cap, dtype=int)]
        clouds.append(local)
    if not clouds:
        raise RuntimeError(f"environment {environment_index} has no local map points from any camera")
    return np.ascontiguousarray(np.concatenate(clouds, axis=0), dtype=np.float32), len(clouds)


def main() -> None:
    if ARGS.num_envs <= 0 or ARGS.trajectory_steps <= 1 or ARGS.frames_per_step <= 0 or ARGS.points_per_camera <= 0:
        raise ValueError("num-envs, trajectory-steps, frames-per-step and points-per-camera must be positive")

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    terrain_cfg, layouts = _make_batched_terrain_cfg(ARGS.terrain, ARGS.seed, ARGS.num_envs)
    env_cfg = parse_env_cfg(TASK, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.seed = ARGS.seed
    env_cfg.terrain.terrain_generator = terrain_cfg
    env_cfg.terrain.max_init_terrain_level = ARGS.num_envs - 1
    env_cfg.terrain.num_envs = ARGS.num_envs
    env_cfg.episode_length_s = 60.0
    env = RslRlVecEnvWrapper(VectorizedAnymalCaptureEnv(env_cfg), clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw_env = env.unwrapped
    camera_count = sum(len(camera.data.pos_w) for camera in raw_env._capture_cameras.values())
    expected_camera_count = required_camera_count(ARGS.num_envs)
    if camera_count != expected_camera_count or len(layouts) != ARGS.num_envs:
        raise RuntimeError(
            f"parallel capture alignment failed: cameras={camera_count}/{expected_camera_count}, layouts={len(layouts)}/{ARGS.num_envs}"
        )

    # TerrainImporter randomly assigns levels at construction.  This probe makes
    # environment i use tile i, so its camera data and dense geometry stay paired.
    env_ids = torch.arange(ARGS.num_envs, device=raw_env.device, dtype=torch.long)
    raw_env._terrain.terrain_levels = env_ids.clone()
    raw_env._terrain.terrain_types = torch.zeros_like(env_ids)
    raw_env._terrain.env_origins = raw_env._terrain.terrain_origins[env_ids, 0].clone()
    raw_env._reset_idx(env_ids)
    # The training environment randomizes episode_length_buf on a full reset to
    # decorrelate resets.  That is useful for RL, but invalid for capture: it
    # can make a trajectory time out during the settle phase.  Capture always
    # starts a fresh episode clock.
    raw_env.episode_length_buf[:] = 0

    motion_rng = np.random.default_rng(ARGS.motion_seed if ARGS.motion_seed is not None else ARGS.seed + 1_000_003)
    if ARGS.random_motion:
        motion_commands = np.column_stack(
            (
                motion_rng.uniform(ARGS.forward_velocity_min, ARGS.forward_velocity_max, size=ARGS.num_envs),
                motion_rng.uniform(ARGS.lateral_velocity_min, ARGS.lateral_velocity_max, size=ARGS.num_envs),
                motion_rng.uniform(ARGS.yaw_rate_min, ARGS.yaw_rate_max, size=ARGS.num_envs),
            )
        ).astype(np.float32)
        initial_yaws = motion_rng.uniform(-np.pi / 6.0, np.pi / 6.0, size=ARGS.num_envs).astype(np.float32)
        default_root_state = raw_env._robot.data.default_root_state.clone()
        default_root_state[:, :3] += raw_env._terrain.env_origins
        default_root_state[:, 3:7] = _yaw_quaternions(initial_yaws, raw_env.device)
        raw_env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    else:
        motion_commands = np.tile(np.array((0.8, 0.0, 0.0), dtype=np.float32), (ARGS.num_envs, 1))
        initial_yaws = np.array(
            [_yaw_from_quaternion(q) for q in raw_env._robot.data.root_quat_w.detach().cpu().numpy()], dtype=np.float32
        )
    raw_env._commands[:] = torch.as_tensor(motion_commands, dtype=torch.float32, device=raw_env.device)
    obs = env.get_observations()
    _set_all_camera_poses(raw_env)

    alive = np.ones(ARGS.num_envs, dtype=bool)
    termination_causes: list[list[str]] = [[] for _ in range(ARGS.num_envs)]

    def _update_alive(dones: torch.Tensor) -> None:
        """Record the first termination cause for each live environment."""
        done_values = dones.detach().cpu().numpy().astype(bool)
        terminated = raw_env.reset_terminated.detach().cpu().numpy().astype(bool)
        timeouts = raw_env.reset_time_outs.detach().cpu().numpy().astype(bool)
        for index in np.flatnonzero(done_values & alive):
            causes = []
            if terminated[index]:
                causes.append("base_contact")
            if timeouts[index]:
                causes.append("time_out")
            termination_causes[index].append("+".join(causes) if causes else "done_unspecified")
        alive[:] &= ~done_values

    for _ in range(ARGS.settle_env_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        _update_alive(dones)
        _set_all_camera_poses(raw_env)

    terrain_origins = raw_env._terrain.terrain_origins.detach().cpu().numpy()
    ground_z = raw_env._terrain.env_origins[:, 2].detach().cpu().numpy()
    structures_by_environment = [
        _translate_structures(layouts[index], terrain_origins[index, 0, :2]) for index in range(ARGS.num_envs)
    ]
    previous_base = raw_env._robot.data.root_pos_w.detach().cpu().numpy().copy()
    previous_yaw = np.array([_yaw_from_quaternion(q) for q in raw_env._robot.data.root_quat_w.detach().cpu().numpy()])
    payloads: list[dict[str, np.ndarray | str]] = [dict() for _ in range(ARGS.num_envs)]
    captured = np.zeros(ARGS.num_envs, dtype=int)
    visible_camera_counts: list[list[int]] = [[] for _ in range(ARGS.num_envs)]

    for frame in range(ARGS.trajectory_steps):
        for _ in range(ARGS.frames_per_step):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            _update_alive(dones)
            _set_all_camera_poses(raw_env)
        bases = raw_env._robot.data.root_pos_w.detach().cpu().numpy()
        yaws = np.array([_yaw_from_quaternion(q) for q in raw_env._robot.data.root_quat_w.detach().cpu().numpy()])
        for index in np.flatnonzero(alive):
            measurement, visible_camera_count = _depth_measurement(
                raw_env._capture_cameras, index, bases[index, :2], yaws[index], raw_env.device, ARGS.points_per_camera
            )
            target = _dense_target(structures_by_environment[index], float(ground_z[index]), bases[index, :2], yaws[index])
            world_translation = bases[index, :2] - previous_base[index, :2]
            inverse_previous_yaw = np.array(
                ((np.cos(previous_yaw[index]), np.sin(previous_yaw[index])), (-np.sin(previous_yaw[index]), np.cos(previous_yaw[index])))
            )
            translation = inverse_previous_yaw @ world_translation
            suffix = f"{captured[index]:02d}"
            payloads[index][f"measurement_{suffix}"] = measurement
            payloads[index][f"target_{suffix}"] = target
            payloads[index][f"translation_{suffix}"] = np.array((translation[0], translation[1], 0.0), dtype=np.float32)
            payloads[index][f"yaw_{suffix}"] = np.array(float(yaws[index] - previous_yaw[index]), dtype=np.float32)
            captured[index] += 1
            visible_camera_counts[index].append(visible_camera_count)
            previous_base[index] = bases[index]
            previous_yaw[index] = yaws[index]
        print(f"[vectorized] frame={frame:02d} alive={alive.tolist()} captured={captured.tolist()}", flush=True)

    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index in range(ARGS.num_envs):
        if captured[index] < 2:
            print(f"[SKIP] environment {index} captured only {captured[index]} frames", flush=True)
            continue
        payloads[index]["metadata_json"] = json.dumps(
            {
                "capture_schema_version": REPRODUCTION_CAPTURE_SCHEMA_VERSION,
                "coordinate_frame": "robot_centric_local_map",
                "map_size_m": MAP_SIZE,
                "voxel_size_m": PAPER_VOXEL_SIZE_M,
                "grid_size": PAPER_GRID_SIZE,
                "trajectory_steps": int(captured[index]),
                "scene_seed": layout_seed_for_environment(ARGS.seed, index),
                "base_seed": ARGS.seed,
                "environment_index": index,
                "motion_randomized": bool(ARGS.random_motion),
                "motion_seed": int(ARGS.motion_seed if ARGS.motion_seed is not None else ARGS.seed + 1_000_003),
                "command_xyz": motion_commands[index].tolist(),
                "yaw_rate_range": [float(ARGS.yaw_rate_min), float(ARGS.yaw_rate_max)],
                "forward_velocity_range": [float(ARGS.forward_velocity_min), float(ARGS.forward_velocity_max)],
                "lateral_velocity_range": [float(ARGS.lateral_velocity_min), float(ARGS.lateral_velocity_max)],
                "initial_yaw": float(initial_yaws[index]),
                "termination_causes": termination_causes[index],
                "camera_count": len(CAMERA_DIRECTIONS),
                "camera_tilt_degrees": PAPER_CAMERA_TILT_DEGREES,
                "visible_camera_count_per_frame": visible_camera_counts[index],
                "points_per_camera": ARGS.points_per_camera,
                "terrain": ARGS.terrain,
                "ground_truth": "dense samples from known primitive geometry; see terrain_profile in provenance",
                "drive": "official Isaac-Velocity-Rough-Anymal-C-Direct-v0 checkpoint",
                "collector": "vectorized_camera_prototype",
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
            }
        )
        output = output_path_for_environment(output_dir, ARGS.terrain, ARGS.seed, index)
        np.savez(output, **payloads[index])
        print(f"[SAVED] {output} ({captured[index]} frames)", flush=True)
        saved += 1
    if saved == 0:
        raise RuntimeError("no parallel environment yielded a usable two-frame trajectory")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        # ``os._exit`` below intentionally bypasses Python's normal unhandled-
        # exception reporting.  Print the traceback here first, otherwise a
        # failed headless probe becomes impossible to diagnose from its log.
        import traceback

        exit_code = 1
        traceback.print_exc()
    finally:
        # Mirrors the serial collector: a completed headless Isaac Sim process
        # must not remain alive and reserve GPU memory after writing NPZ files.
        import os

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
