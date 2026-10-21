"""Q1 flat rolling with explicit posture, support, parking and path quality."""
from wheel_legged_gym.envs.q1_flat.q1_flat_config import Q1FlatCfg, Q1FlatCfgPPO


class Q1Cfg(Q1FlatCfg):
    class env(Q1FlatCfg.env):
        num_envs = 4096
        num_observations = 31  # original 27 + parking XY, cross-track, heading
        num_privileged_obs = 151  # original layout + 4 actor + 4 wheel support

    class commands(Q1FlatCfg.commands):
        stand_probability = 0.25
        minimum_motion_speed = 0.15
        endpoint_probability = 0.5
        acceleration_limit = 0.4  # m/s^2; applies in train and play
        idle_threshold = 0.02
        parking_grace_s = 0.5
        parking_ramp_s = 0.5

    class control(Q1FlatCfg.control):
        damping = {"thigh_joint": 1.0, "calf_joint": 1.0, "foot_joint": 0.20}
        wheel_velocity_feedforward = True  # command vx/r + learned balance residual

    class rewards(Q1FlatCfg.rewards):
        tracking_sigma = 0.0064
        idle_tracking_sigma = 0.0025
        yaw_tracking_sigma = 0.04
        height_sigma = 0.0009
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
            tracking_ang_vel = 0.4
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
        run_name = "low_speed"
        max_iterations = 30000
        save_interval = 500
