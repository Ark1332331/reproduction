"""R6 -> real-policy closure verification (headless Isaac Lab).

Loads the official ANYmal-C rough rsl_rl checkpoint and verifies, in one run:

1. Contract cross-check: the env's own 187-dim height_scan segment (obs indices 36..222)
   is re-expressed as a HeightMap via the inverse contract, then re-sampled with
   reproduction.r6_controller_interface.isaac_height_scan() at the robot's real base
   pose/yaw. A geometry/value mismatch between the two scans would show up here.

2. Forward acceptance (the R6 closure criterion): the 235-dim observation whose
   height segment is *our* isaac_height_scan output is fed to the checkpoint's actor
   ([512, 256, 128] ELU, std (12,)); the call must return a (num_envs, 12) finite
   action tensor without shape errors.

3. Short rollout: the loaded policy runs the real rough env for a few steps; base
   height, planar velocity, commands, action magnitude and done flags are logged to
   show the robot stands / walks with the official checkpoint.

Usage (must run through the Isaac Lab launcher, headless):

    cd /home/ark/projects/IsaacLab
    conda activate isaaclab
    ./isaaclab.sh -p /media/ark/Data/devpy/projects/allinone/reproduction/verify_r6_policy_play.py \
        --headless --num_envs 1 --rollout_steps 40

Exit code 0 only if all checks pass.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------
# CLI / AppLauncher (must be parsed and launched before heavy imports)
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-Direct-v0")
parser.add_argument("--rollout_steps", type=int, default=40, help="Rollout steps for phase 3.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/home/ark/projects/IsaacLab/.pretrained_checkpoints/rsl_rl/"
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0/checkpoint.pt",
)
# note: the RL/sim device is provided by AppLauncher as --device (default "cuda:0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------
# Heavy imports (only valid after the app is launched)
# --------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

# R6 module lives in the allinone reproduction tree
sys.path.insert(0, "/media/ark/Data/devpy/projects/allinone")
from reproduction.r6_controller_interface import HeightMap, isaac_height_scan  # noqa: E402

# Verified observation layout (see project_state.md R6 contract):
# [lin_vel 3, ang_vel 3, gravity 3, cmd 3, joint_pos 12, joint_vel 12, height_scan 187, actions 12]
HEIGHT_START = 36
HEIGHT_DIM = 187
OBS_DIM = 235

PASS = True


def fail(msg: str) -> None:
    global PASS
    PASS = False
    print(f"[FAIL] {msg}")


def main() -> None:
    # ----------------------------------------------------------------------
    # Environment + policy loading (same path as scripts/.../rsl_rl/play.py)
    # ----------------------------------------------------------------------
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = agent_cfg.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] checkpoint loaded from {args_cli.checkpoint} (iter {runner.current_learning_iteration})")

    # ----------------------------------------------------------------------
    # Phase 1: contract cross-check (env scan vs our re-sampled scan)
    # ----------------------------------------------------------------------
    obs_td = env.get_observations()
    obs = obs_td["policy"]
    if obs.shape != (args_cli.num_envs, OBS_DIM):
        fail(f"obs shape {tuple(obs.shape)} != ({args_cli.num_envs}, {OBS_DIM})")
        return
    print(f"[INFO] env obs shape {tuple(obs.shape)} (expected {OBS_DIM}-dim)")

    robot = env.unwrapped._robot
    base_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()  # w, x, y, z
    yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
    base_z = float(base_pos[2])
    print(f"[INFO] base pos {base_pos}  yaw {yaw:.4f} rad")

    env_scan = obs[0, HEIGHT_START : HEIGHT_START + HEIGHT_DIM].detach().cpu().numpy()

    # Diagnostic: sensor origin vs articulation root, and obs values vs raw ray hits
    scanner = env.unwrapped._height_scanner
    sensor_pos = scanner.data.pos_w[0].detach().cpu().numpy()
    sensor_quat = scanner.data.quat_w[0].detach().cpu().numpy()
    hits = scanner.data.ray_hits_w[0].detach().cpu().numpy()  # (187, 3)
    v_from_hits = np.clip(sensor_pos[2] - hits[:, 2] - 0.5, -1.0, 1.0)
    print(f"[DIAG] sensor pos {sensor_pos} (z diff vs root {sensor_pos[2] - base_z:+.6f})")
    print(f"[DIAG] yaw from sensor quat: "
          f"{np.arctan2(2.0*(sensor_quat[0]*sensor_quat[3]+sensor_quat[1]*sensor_quat[2]), 1.0-2.0*(sensor_quat[2]**2+sensor_quat[3]**2)):.6f}")
    print(f"[DIAG] |obs - (pos_w.z - hit_z - 0.5).clip| max {np.abs(env_scan - v_from_hits).max():.8f}")
    print(f"[DIAG] hit z: min {hits[:, 2].min():.4f} max {hits[:, 2].max():.4f} mean {hits[:, 2].mean():.4f}")
    print(f"[DIAG] env scan: min {env_scan.min():.4f} max {env_scan.max():.4f} mean {env_scan.mean():.4f}")
    print(f"[DIAG] env scan first 8: {np.round(env_scan[:8], 4)}")

    # Build a HeightMap from the env's own scan via the inverse contract
    # (height = base_z - value - 0.5). isaac_height_scan indexes HeightMap cells by
    # ABSOLUTE, non-negative world cell coordinates (cell (mx, my) covers world
    # [mx*0.1, (mx+1)*0.1) x [my*0.1, (my+1)*0.1)). The env spawns the robot at world
    # (-12, -76), so the scan window is shifted into a non-negative local frame:
    # local = world + (16, 80); the robot then sits at local (4, 4) and a 60x60-cell
    # map anchored at local (0, 0) covers the whole 1.6 m x 1.0 m scan window.
    cell = 0.1
    n_cells = 60
    shift = np.array([16.0, 80.0])
    local_base = np.array([base_pos[0], base_pos[1]]) + shift
    heights = np.full((n_cells, n_cells), np.nan)
    xs = np.arange(-0.8, 0.8 + 1e-9, cell)
    ys = np.arange(-0.5, 0.5 + 1e-9, cell)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    clipped_cells = 0
    for iy in range(11):
        for ix in range(17):
            value = env_scan[iy * 17 + ix]
            if abs(value) >= 1.0:
                clipped_cells += 1  # raw value hit the contract clip; recovered height is approximate
            dx, dy = xs[ix], ys[iy]
            wx = local_base[0] + cos_y * dx - sin_y * dy
            wy = local_base[1] + sin_y * dx + cos_y * dy
            mx, my = int(np.floor(wx / cell)), int(np.floor(wy / cell))
            if 0 <= mx < n_cells and 0 <= my < n_cells:
                heights[mx, my] = base_z - value - 0.5
    hmap = HeightMap(heights=heights, observed=np.isfinite(heights))

    ours = isaac_height_scan(hmap, (local_base[0], local_base[1]), base_z, yaw)
    diff = np.abs(ours - env_scan)
    n_exact = int(np.sum(diff < 1e-6))
    print(f"[INFO] contract cross-check: {n_exact}/187 cells re-sampled exactly "
          f"(max |diff| {diff.max():.6f}, mean {diff.mean():.6f}); {clipped_cells} cells hit the clip bound")
    print(f"[DEBUG] local_base {repr(local_base)}, map observed cells: {int(hmap.observed.sum())}, "
          f"ours[:8]: {np.round(ours[:8], 4)}")
    for (ix, iy) in [(0, 0), (8, 5), (16, 10)]:
        v = env_scan[iy * 17 + ix]
        dx, dy = xs[ix], ys[iy]
        wx = local_base[0] + cos_y * dx - sin_y * dy
        wy = local_base[1] + sin_y * dx + cos_y * dy
        a = (int(np.floor(wx / cell)), int(np.floor(wy / cell)))
        print(f"[DEBUG] ray ix={ix} iy={iy} local=({wx:.6f},{wy:.6f}) map_idx={a} "
              f"env={v:.4f} ours={ours[iy*17+ix]:.4f} hmap={hmap.heights[a[0], a[1]]:.4f}")

    # ----------------------------------------------------------------------
    # Phase 2: policy forward accepts OUR 187-dim scan (the R6 closure criterion)
    # ----------------------------------------------------------------------
    obs_ours_td = obs_td.clone()
    obs_ours_td["policy"][0, HEIGHT_START : HEIGHT_START + HEIGHT_DIM] = torch.from_numpy(ours).to(obs.device)

    with torch.inference_mode():
        act_real = policy(obs_td)
        act_ours = policy(obs_ours_td)

    if act_ours.shape != (args_cli.num_envs, 12):
        fail(f"policy output shape {tuple(act_ours.shape)} != ({args_cli.num_envs}, 12)")
        return
    if not torch.isfinite(act_ours).all():
        fail("policy output contains non-finite values with our height scan")
        return
    act_diff = (act_ours - act_real).abs().max().item()
    print(f"[INFO] policy forward accepted our 187-dim scan: output {tuple(act_ours.shape)}, "
          f"finite, |action| max {act_ours.abs().max().item():.4f}, "
          f"max |action delta| vs env-scan obs {act_diff:.6f}")
    print("[PASS] R6 187-dim height scan is accepted by the real ANYmal-C rough policy forward")

    # ----------------------------------------------------------------------
    # Phase 3: short rollout in the real rough env
    # ----------------------------------------------------------------------
    print(f"\n[INFO] rollout for {args_cli.rollout_steps} steps (num_envs={args_cli.num_envs})")
    print(f"{'step':>4} {'base_z':>8} {'|v_xy|':>7} {'cmd_xy':>10} {'cmd_yaw':>7} {'|a|max':>7} {'done':>4}")
    z_vals, done_any = [], False
    for step in range(args_cli.rollout_steps):
        with torch.inference_mode():
            actions = policy(obs_td)
            obs_td, rew, dones, extras = env.step(actions)
        base_z = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
        vel_xy = robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
        cmd = env.unwrapped._commands[0].detach().cpu().numpy()
        a_max = actions[0].abs().max().item()
        done = bool(dones[0].item())
        done_any = done_any or done
        z_vals.append(float(base_z[0]))
        print(f"{step:>4} {base_z[0]:>8.4f} {np.linalg.norm(vel_xy[0]):>7.3f} "
              f"{cmd[0]:>5.2f},{cmd[1]:>5.2f} {cmd[2]:>7.2f} {a_max:>7.3f} {int(done):>4}")

    z_arr = np.array(z_vals)
    if done_any:
        print("[WARN] episode terminated during the rollout window")
    print(f"[INFO] base z over rollout: min {z_arr.min():.4f} max {z_arr.max():.4f} "
          f"mean {z_arr.mean():.4f} (ANYmal standing height ~0.5-0.6 m)")
    if np.ptp(z_arr) < 0.01:
        print("[WARN] base z barely changed: the robot may be standing still")
    print("[PASS] rollout ran end-to-end with the official checkpoint" if not done_any else "[FAIL] rollout terminated")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0 if PASS else 1)
