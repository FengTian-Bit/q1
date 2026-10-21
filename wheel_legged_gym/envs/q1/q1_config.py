"""Q1 balance, straight travel, in-place turns and moving turns on mixed terrain."""
from wheel_legged_gym.envs.q1_flat.q1_flat_config import Q1FlatCfg, Q1FlatCfgPPO


class Q1Cfg(Q1FlatCfg):
    class env(Q1FlatCfg.env):
        num_envs = 4096
        num_observations = 31  # original 27 + parking XY, cross-track, heading
        num_privileged_obs = 151  # original layout + 4 actor + 4 wheel support

    class terrain(Q1FlatCfg.terrain):
        # Reference: Dreamwaq-highplatform/legged_gym/envs/t0/t0/t0_config.py.
        # Halve slopes/heights, not tread width, tile size or grid resolution.
        mesh_type = "trimesh"
        curriculum = True
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 2  # start on gentle rows, then progress
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        slope_treshold = 0.75
        selected = False
        selected_type = "stairs_up"
        selected_difficulty = 1.0
        # Keep Q1's 77 height probes and the existing network dimensions.
        terrain_names = ["flat", "slope_down", "slope_up", "rough_slope",
                         "stairs_down", "stairs_up", "discrete"]
        terrain_labels = [-1, 0, 0, 1, 2, 3, 4]
        terrain_proportions = [0.10, 0.10, 0.10, 0.15, 0.20, 0.20, 0.15]
        slope_range = [0.05, 0.35]  # rise/run, NOT radians
        rough_slope_range = [0.05, 0.25]
        rough_height_range = [-0.025, 0.025]
        rough_height_step = 0.005
        rough_downsampled_scale = 0.2
        stair_height_range = [0.025, 0.20]  # single riser, both signs
        stair_width = 0.25  # same nominal width as T0; Gym quantizes to cells
        obstacle_height_range = [0.025, 0.15]
        obstacle_size_range = [1.0, 2.0]
        num_obstacles = 20
        platform_size = 3.0
        curriculum_min_duration_s = 10.0
        curriculum_min_distance = 2.0  # leave the central 3 m spawn platform
        curriculum_tracking_error = 0.10

    class commands(Q1FlatCfg.commands):
        stand_probability = 0.25  # flat tiles only
        rough_resampling_time = 20.0
        minimum_motion_speed = 0.15
        endpoint_probability = 0.5
        acceleration_limit = 0.4  # m/s^2; applies in train and play
        yaw_acceleration_limit = 0.8  # rad/s^2; applies in train and play
        minimum_turn_rate = 0.2
        yaw_idle_threshold = 0.05
        # Stand probability is separate; these split the remaining samples.
        straight_probability = 1.0/3.0
        in_place_probability = 1.0/3.0
        idle_threshold = 0.02
        parking_grace_s = 0.5
        parking_ramp_s = 0.5

        class ranges(Q1FlatCfg.commands.ranges):
            ang_vel_yaw = [-0.6, 0.6]

    class asset(Q1FlatCfg.asset):
        # Wheel collision center Y in the trunk frame, from Q1 URDF chains.
        wheel_lateral_offsets = [0.14927, -0.14973]

    class control(Q1FlatCfg.control):
        damping = {"thigh_joint": 1.0, "calf_joint": 1.0, "foot_joint": 0.20}
        wheel_velocity_feedforward = True  # command vx/r + learned balance residual

    class rewards(Q1FlatCfg.rewards):
        tracking_sigma = 0.0064
        idle_tracking_sigma = 0.0025
        yaw_tracking_sigma = 0.04
        height_sigma = 0.0009
        rough_height_sigma = 0.01
        pose_allowance = 0.08
        pose_error_scale = 0.20
        symmetry_allowance = 0.08
        support_force = 5.0  # contact threshold, not a fixed equal-load target
        clearance_allowance = 0.015
        clearance_scale = 0.03
        wheel_width = 0.02999
        wheel_center_y = [-0.015, 0.015]
        slip_deadband = 0.03
        slip_scale = 0.20
        parking_position_scale = 0.15
        path_position_scale = 0.20
        heading_scale = 0.20
        wheel_idle_deadband = 0.25  # rad/s, permits balance corrections
        wheel_idle_scale = 2.0

        class scales(Q1FlatCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            turning_yaw_tracking = 0.5
            base_height = 0.8
            orientation = -2.5
            actual_posture = -0.25
            leg_symmetry = -0.10
            rolling_contact = -0.30
            rolling_leg_motion = -0.02
            rolling_slip = -0.10
            speed_error = -0.40
            parking_drift = -0.50
            parking_wheel_speed = -0.04
            straight_path = -0.20


class Q1CfgPPO(Q1FlatCfgPPO):
    class policy(Q1FlatCfgPPO.policy):
        num_encoder_obs = 155  # 31 x 5

    class runner(Q1FlatCfgPPO.runner):
        experiment_name = "q1"
        run_name = "turning"
        max_iterations = 30000
        save_interval = 500


def configure_q1_stage(cfg, stage):
    """Apply a stage explicitly, including when reusing a registry config."""
    if stage not in ("balance", "low_speed", "turning"):
        raise ValueError("Unknown Q1 stage: {}".format(stage))
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0] if stage == "balance" else [-0.3, 0.3]
    cfg.commands.ranges.ang_vel_yaw = [-0.6, 0.6] if stage == "turning" else [0.0, 0.0]
    cfg.commands.stand_probability = 1.0 if stage == "balance" else 0.25
