"""Q1 open-chain joint PD baseline: balance and slow longitudinal motion."""

from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
)


class Q1FlatCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_envs = 1024
        num_observations = 27
        # Keep the base asymmetric critic and five-frame velocity estimator.
        num_privileged_obs = 143
        obs_history_length = 5
        obs_history_dec = 1
        num_actions = 6
        episode_length_s = 20.0
        fail_to_terminal_time_s = 0.0
        # Use the simulator velocity, avoiding a spike when a wheel angle wraps.
        dof_vel_use_pos_diff = False
        max_tilt_rad = 0.75
        min_base_height = 0.20

    class terrain(WheelLeggedCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True  # 77 zero heights keep the inherited critic layout.
        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0

    class commands(WheelLeggedCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 4.0
        stand_probability = 0.5

        class ranges:
            lin_vel_x = [-0.3, 0.3]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.397, 0.397]
            heading = [0.0, 0.0]

    class init_state(WheelLeggedCfg.init_state):
        # Exact primitive FK gives a wheel bottom of 5.34 mm above the plane.
        pos = [0.0, 0.0, 0.402]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        reset_lin_vel = 0.02  # horizontal perturbation, m/s
        reset_ang_vel = 0.03  # roll/pitch perturbation, rad/s
        default_joint_angles = {
            "L_thigh_joint": 0.6,
            "L_calf_joint": -1.2,
            "L_foot_joint": 0.0,
            "R_thigh_joint": 0.6,
            "R_calf_joint": -1.2,
            "R_foot_joint": 0.0,
        }

    class control(WheelLeggedCfg.control):
        pos_action_scale = 0.35  # rad per normalized leg action
        vel_action_scale = 10.0  # rad/s per normalized wheel action
        stiffness = {"thigh_joint": 40.0, "calf_joint": 40.0, "foot_joint": 0.0}
        damping = {"thigh_joint": 1.0, "calf_joint": 1.0, "foot_joint": 0.10}
        decimation = 2  # 100 Hz policy / 200 Hz physics

    class asset(WheelLeggedCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/q1/urdf/q1.urdf"
        name = "Q1"
        # Policy order is independent of the importer DOF order.
        policy_joint_names = [
            "L_thigh_joint", "L_calf_joint", "L_foot_joint",
            "R_thigh_joint", "R_calf_joint", "R_foot_joint",
        ]
        wheel_body_names = ["L_foot", "R_foot"]
        foot_name = "_foot"
        offset = 0.0
        l1 = 0.2
        l2 = 0.2
        wheel_radius = 0.07496
        penalize_contacts_on = ["trunk", "thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 0
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(WheelLeggedCfg.domain_rand):
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        randomize_inertia = False
        randomize_base_com = False
        push_robots = False
        randomize_Kp = False
        randomize_Kd = False
        randomize_motor_torque = False
        randomize_default_dof_pos = False
        randomize_action_delay = False
        delay_ms_range = [0, 0]

    class rewards(WheelLeggedCfg.rewards):
        only_positive_rewards = False
        clip_single_reward = 10.0
        tracking_sigma = 0.04
        base_height_target = 0.397
        height_sigma = 0.0025

        class scales:
            alive = 1.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.3
            base_height = 0.5
            nominal_state = -0.1
            lateral_velocity = -0.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -2.0
            dof_vel = -5e-5
            dof_acc = -1e-7
            torques = -1e-4
            action_rate = -0.005
            action_smooth = -0.005
            collision = -1.0
            dof_pos_limits = -2.0
            termination = -2.0

    class normalization(WheelLeggedCfg.normalization):
        clip_actions = 1.0

    class noise(WheelLeggedCfg.noise):
        add_noise = False


class Q1FlatCfgPPO(WheelLeggedCfgPPO):
    class policy(WheelLeggedCfgPPO.policy):
        num_encoder_obs = 135
        init_noise_std = 0.5

    class algorithm(WheelLeggedCfgPPO.algorithm):
        learning_rate = 5e-4
        desired_kl = 0.01

    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "q1_flat"
        run_name = "low_speed"
        max_iterations = 3000
        save_interval = 100
