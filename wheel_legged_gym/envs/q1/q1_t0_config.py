"""Q1 contact-guided control on the ordinary four-legged T0 terrain course.

Q1 uses the T0 terrain primitives with its own severity and command ranges.
It keeps its own actuators, observations and CTBC controller.
"""
from copy import deepcopy

from wheel_legged_gym.envs.q1_ctbc.q1_ctbc_config import (
    Q1CTBCCfg, Q1CTBCCfgPPO, configure_ctbc_stage,
)


T0_TERRAIN = {
    "generation_mode": "t0", "mesh_type": "trimesh", "curriculum": True,
    "horizontal_scale": 0.1, "vertical_scale": 0.005, "border_size": 25.0,
    "terrain_length": 8.0, "terrain_width": 8.0, "terrain_spacing": 0.0,
    "num_rows": 20, "num_cols": 20, "max_init_terrain_level": 5,
    "static_friction": 1.0, "dynamic_friction": 1.0, "restitution": 0.0,
    "slope_treshold": 0.75, "difficulty_range": [0.0, 1.0],
    "terrain_names": ["flat", "slope_down", "slope_up", "rough_slope_down",
                      "rough_slope_up", "stairs_down", "stairs_up"],
    "terrain_labels": [-1, 0, 0, 1, 1, 2, 3],
    "terrain_proportions": [0.20, 0.15, 0.15, 0.05, 0.05, 0.20, 0.20],
    "slope_range": [0.1, 0.48], "rough_slope_range": [0.1, 0.32],
    "rough_height_range": [-0.05, 0.05], "rough_height_step": 0.005,
    "rough_downsampled_scale": 0.2, "stair_height_range": [0.05, 0.22],
    "stair_width": 0.31, "platform_size": 3.0,
    "obstacle_height_range": [0.05, 0.30], "obstacle_size_range": [1.0, 2.0],
    "num_obstacles": 20, "selected": False, "selected_type": "stairs_down",
    "selected_difficulty": 0.5,
}


def configure_q1_t0(cfg):
    """Set the new task preset, including when migrating a CTBC checkpoint."""
    configure_ctbc_stage(cfg, "stairs20", high_stairs_profile="legacy20")
    for name, value in T0_TERRAIN.items():
        setattr(cfg.terrain, name, deepcopy(value))
    cfg.ctbc.stage = "t0"
    cfg.env.episode_length_s = 20.0
    cfg.commands.curriculum = False  # T0 command-range curriculum is disabled.
    cfg.commands.heading_command = True
    cfg.commands.resampling_time = 10.0
    cfg.commands.rough_resampling_time = 10.0
    cfg.commands.max_forward_speed = 1.2
    cfg.commands.minimum_motion_speed = 0.2
    cfg.commands.ranges.lin_vel_x = [-1.2, 1.2]
    cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    cfg.commands.ranges.heading = [-3.14, 3.14]
    # Q1 has no lateral command channel: [vx, yaw_rate, body_height].
    # No terminal platform or world-X progress exists on T0 pyramids.
    cfg.rewards.scales.ctbc_progress = 0.0
    cfg.rewards.scales.ctbc_success = 0.0
    cfg.rewards.scales.ctbc_same_x = 0.0
    return cfg


class Q1T0Cfg(Q1CTBCCfg):
    class rewards(Q1CTBCCfg.rewards):
        # Locked q1_tcn reward v5; CTBC-only teacher terms stay inherited.
        reward_version = 5
        tracking_sigma = 0.0025
        idle_tracking_sigma = 0.0004
        yaw_tracking_sigma = tracking_sigma
        height_sigma = 0.0025
        parking_speed_threshold = 0.02
        parking_yaw_threshold = 0.05
        parking_brake_s = 0.5
        parking_ramp_s = 0.5
        parking_deadband_m = 0.005
        parking_scale_m = 0.03
        parking_velocity_scale = 0.02
        parking_velocity_deadband = 0.001

        class scales(Q1CTBCCfg.rewards.scales):
            alive = 0.0
            nominal_state = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            action_smooth = 0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            parking_drift = -3.0
            parking_velocity = -0.5
            orientation = -20.0
            base_height = 0.5
            lateral_velocity = -0.5
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            action_rate = -0.02
            torques = -0.0001
            collision = -1.0
            dof_pos_limits = -2.0
            termination = -200.0
            # Replace the old locomotion objective rather than summing it
            # with v5 (in particular, avoid double-counting yaw tracking).
            turning_yaw_tracking = 0.0
            actual_posture = 0.0
            leg_symmetry = 0.0
            rolling_contact = 0.0
            rolling_leg_motion = 0.0
            rolling_slip = 0.0
            speed_error = 0.0
            parking_wheel_speed = 0.0
            straight_path = 0.0

    class env(Q1CTBCCfg.env):
        # T0 keeps its existing batch when CTBC defaults follow the saved run.
        num_envs = 4096

    def __init__(self):
        super().__init__()
        configure_q1_t0(self)


class Q1T0CfgPPO(Q1CTBCCfgPPO):
    class policy(Q1CTBCCfgPPO.policy):
        velocity_estimation_enabled = False

    class runner(Q1CTBCCfgPPO.runner):
        experiment_name = "q1"
        run_name = "t0"
        swanlab_tags = ["q1", "ctbc", "t0-terrain"]
