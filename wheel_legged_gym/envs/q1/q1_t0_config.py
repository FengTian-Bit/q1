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
    class env(Q1CTBCCfg.env):
        # T0 keeps its existing batch when CTBC defaults follow the saved run.
        num_envs = 4096

    def __init__(self):
        super().__init__()
        configure_q1_t0(self)


class Q1T0CfgPPO(Q1CTBCCfgPPO):
    class runner(Q1CTBCCfgPPO.runner):
        experiment_name = "q1"
        run_name = "t0"
        swanlab_tags = ["q1", "ctbc", "t0-terrain"]
