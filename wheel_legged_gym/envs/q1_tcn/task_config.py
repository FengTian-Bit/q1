"""Fixed v7 task: random initialization, jointly trained parking and motion.

The experimental Q1TCNCfg classes remain unchanged. Only the registered public
q1_tcn task uses these settings; no checkpoint or pilot JSON is required.
"""
from .q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO, validate_config


PROFILE_VERSION = "q1_tcn_v7_joint_v1"


class Q1TCNTaskCfg(Q1TCNCfg):
    class env(Q1TCNCfg.env):
        num_envs = 4096
        num_observations = 30
        num_privileged_obs = 146
        episode_length_s = 24.0

    class commands(Q1TCNCfg.commands):
        target_mode = True
        target_stage = "turning"
        sequence_fraction = 0.75
        sequence_min_fraction = 0.2
        target_edge_fraction = 0.25
        parking_env_fraction = 0.2
        stand_probability = 0.2
        max_linear_acceleration = 2.0

        class ranges(Q1TCNCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]
            ang_vel_yaw = [-1.0, 1.0]

    class terrain(Q1TCNCfg.terrain):
        static_friction = 0.8
        dynamic_friction = 0.8

    class domain_rand(Q1TCNCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_base_com = False
        push_robots = False

    class odometry(Q1TCNCfg.odometry):
        enabled = True

    class normalization(Q1TCNCfg.normalization):
        clip_actions = 3.0
        leg_clip_actions = 1.0

    class rewards(Q1TCNCfg.rewards):
        reward_version = 7
        tracking_sigma = 0.04
        idle_tracking_sigma = 0.0004
        yaw_tracking_sigma = 0.09
        parking_deadband_m = 0.005
        parking_scale_m = 0.03
        relative_tracking = dict(fraction=0.1, linear_floor=0.03,
                                 yaw_floor=0.05, zero_yaw_width=0.1)
        wheel_diff_tracking = dict(track_width=0.299, sigma=1.0)

        class scales(Q1TCNCfg.rewards.scales):
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            tracking_wheel_diff = 0.5
            parking_drift = -1.0
            parking_velocity = -0.2
            orientation = -4.0
            ang_vel_xy = -0.05
            action_rate = -0.005


class Q1TCNTaskCfgPPO(Q1TCNCfgPPO):
    ablation = "A0"

    class policy(Q1TCNCfgPPO.policy):
        mean_limit = 0.95
        init_noise_std = 0.08
        max_noise_std = 0.15
        learn_noise = False
        wheel_residual = True
        wheel_residual_limit = 0.3
        wheel_residual_yaw_scale = 4.0
        parking_policy = True
        parking_blend = True
        freeze_motion_backbone = False
        command_reference = dict(limit=0.375, command_scale=2.0,
            wheel_radius=0.07496, wheel_action_scale=10.0,
            wheel_velocity_obs_scale=0.05, action_limit=3.0, feedforward_gain=1.5)
        braking_wheel_extension = dict(start=0.85, slope=1.0, limit=3.0,
            moving_limit=3.0, moving_gravity_start=0.08, moving_gravity_full=0.14)
        steering_feedforward = dict(command_scale=0.25, wheel_y=[0.14927, -0.14973],
            wheel_radius=0.07496, wheel_action_scale=10.0, gain=1.5)
        motion_wheel_gain = dict(start_speed=0.6, full_speed=1.2, gain=2.0, command_scale=2.0)

    class algorithm(Q1TCNCfgPPO.algorithm):
        learning_rate = 3e-5
        desired_kl = None
        max_kl = 0.02
        num_learning_epochs = 3
        entropy_coef = 0.0001
        gamma = 0.998
        symmetry_loss_coef = 0.1
        symmetry_batch_size = 4096

    class runner(Q1TCNCfgPPO.runner):
        experiment_name = "q1_tcn"
        run_name = "v7_joint"
        max_iterations = 30000
        resume = False
        load_run = -1
        checkpoint = -1
        logger = "swanlab"
        swanlab_project = "Q1"
        swanlab_mode = "online"


def validate_task_config(cfg, train, training=True):
    validate_config(cfg, train)
    if not 0 < cfg.commands.parking_env_fraction < 1:
        raise ValueError("Joint training requires a parking_env_fraction between 0 and 1")
    if training and cfg.env.num_envs < 2:
        raise ValueError("Joint training needs at least two environments")
    if train.ablation != "A0" or not train.policy.parking_policy or not train.policy.parking_blend:
        raise ValueError("The fixed task requires A0 and the blended parking branch")
    if train.policy.freeze_motion_backbone:
        raise ValueError("From-scratch joint training must train the motion backbone")
