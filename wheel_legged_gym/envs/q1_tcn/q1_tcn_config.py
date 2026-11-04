"""Shared training conditions for the A0--A4 state-estimation ablation."""

from wheel_legged_gym.envs.q1_flat.q1_flat_config import Q1FlatCfg
from wheel_legged_gym.envs.base.base_config import BaseConfig


ABLATIONS = {
    "A0": (False, False, False),
    "A1": (True, False, False),
    "A2": (True, True, False),
    "A3": (True, False, True),
    "A4": (True, True, True),
}


class Q1TCNCfg(Q1FlatCfg):
    seed = 1

    class env(Q1FlatCfg.env):
        obs_history_length = 16

    class domain_rand(Q1FlatCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.35, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.2, 0.5]
        randomize_base_com = True
        rand_com_vec = [0.01, 0.005, 0.005]
        # Q1TCN replaces inherited impulsive pushes with finite force pulses.
        push_robots = True
        push_interval_s = 5.0
        push_interval = 500  # derived by LeggedRobot._parse_cfg, retained in snapshots
        push_duration_s = 0.15
        push_force_xy = 5.0  # N per horizontal axis, applied at trunk COM

    class noise(Q1FlatCfg.noise):
        add_noise = True

    class rewards(Q1FlatCfg.rewards):
        reward_version = 2
        tracking_sigma = 0.01
        yaw_tracking_sigma = 0.04
        parking_speed_threshold = 0.02
        parking_yaw_threshold = 0.05
        parking_brake_s = 0.5
        parking_ramp_s = 0.5
        parking_deadband_m = 0.02
        parking_scale_m = 0.10

        class scales(Q1FlatCfg.rewards.scales):
            alive = 0.0
            nominal_state = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            action_smooth = 0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.3
            parking_drift = -2.0
            orientation = -2.0
            base_height = 0.5
            lateral_velocity = -0.5
            lin_vel_z = -0.2
            ang_vel_xy = -0.05
            action_rate = -0.01
            torques = -0.0001
            collision = -1.0
            dof_pos_limits = -2.0
            termination = -200.0  # multiplied by dt: -2 per failure at 100 Hz

    class estimator:
        contact_on_force = 5.0  # N, world Z on the flat plane
        contact_off_force = 3.0
        contact_confirm_frames = 2  # policy frames, 20 ms at 100 Hz
        velocity_scale = 1.0  # m/s per normalized target unit
        slip_scale = 1.0  # m/s per normalized target unit
        wheel_center_y = [-0.015, 0.015]  # Q1 cylinder collision origins
        wheel_width = 0.02999
        payload_com_offset_x = 0.0  # m; deterministic evaluation-only offset


class Q1TCNCfgPPO(BaseConfig):
    seed = 1
    ablation = "A1"

    class policy:
        history_length = 16
        channels = 64
        dilations = [1, 2, 4, 8]
        kernel_size = 2
        latent_dim = 16
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.5

    class algorithm:
        learning_rate = 5e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.01
        max_grad_norm = 1.0
        desired_kl = 0.01
        velocity_loss_coef = 1.0
        contact_loss_coef = 1.0
        slip_loss_coef = 1.0

    class runner:
        num_steps_per_env = 48
        max_iterations = 3000
        save_interval = 100
        experiment_name = "q1_tcn_reward_v2"


def validate_config(env_cfg, train_cfg):
    if train_cfg.ablation not in ABLATIONS:
        raise ValueError("ablation must be one of A0, A1, A2, A3, A4")
    if train_cfg.policy.history_length != env_cfg.env.obs_history_length:
        raise ValueError("TCN history_length must match env.obs_history_length")
    if env_cfg.terrain.mesh_type != "plane":
        raise ValueError("q1_tcn labels currently support a flat plane only")
    if env_cfg.estimator.velocity_scale <= 0 or env_cfg.estimator.slip_scale <= 0:
        raise ValueError("Estimator normalization scales must be positive")
    if env_cfg.estimator.contact_on_force <= env_cfg.estimator.contact_off_force:
        raise ValueError("Contact hysteresis requires on_force > off_force")
    if env_cfg.estimator.contact_confirm_frames < 1:
        raise ValueError("contact_confirm_frames must be positive")
    pulse = env_cfg.domain_rand
    if not 0 < pulse.push_duration_s < pulse.push_interval_s:
        raise ValueError("push duration must be positive and less than interval")
    rewards = env_cfg.rewards
    if rewards.parking_brake_s < 0 or rewards.parking_ramp_s <= 0 or rewards.parking_scale_m <= 0:
        raise ValueError("Parking timing and distance scales must be valid")
