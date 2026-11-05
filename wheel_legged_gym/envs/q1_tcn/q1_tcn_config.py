"""Shared training conditions for the A0--A4 state-estimation ablation."""

import math
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

    class commands(Q1FlatCfg.commands):
        sequence_fraction = 0.0  # optional stop/drive/reverse command coverage
        sequence_min_fraction = 0.85  # fraction of current range in sequence trials
        target_mode = False  # fixed full-range training, no speed curriculum
        target_stage = 'linear'
        target_edge_fraction = 0.25
        max_linear_acceleration = 0.0  # m/s^2; zero preserves legacy step commands

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

    class normalization(Q1FlatCfg.normalization):
        leg_clip_actions = None  # optional separate leg bound when opening wheel headroom

    class rewards(Q1FlatCfg.rewards):
        reward_version = 2
        tracking_sigma = 0.01
        idle_tracking_sigma = 0.01
        yaw_tracking_sigma = 0.04
        relative_tracking = {}  # reward v7: moving-command error widths
        wheel_diff_tracking = {}  # optional track_width(m), sigma((rad/s)^2)
        parking_speed_threshold = 0.02
        parking_yaw_threshold = 0.05
        parking_brake_s = 0.5
        parking_ramp_s = 0.5
        parking_deadband_m = 0.02
        parking_scale_m = 0.10
        parking_velocity_scale = 0.02
        parking_velocity_deadband = 0.001

        class scales(Q1FlatCfg.rewards.scales):
            alive = 0.0
            nominal_state = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            action_smooth = 0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.3
            tracking_wheel_diff = 0.0  # auxiliary guidance; body yaw remains the target
            parking_drift = -2.0
            parking_velocity = 0.0
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

    class odometry:
        enabled = False
        position_scale = 0.05
        gravity_gain = 0.02
        observe_wheel_increments = False
        increment_scale = 0.1  # rad per normalized encoder increment
        parking_observation_delay_s = 0.0  # optional post-motion handoff delay; initial standing stays immediate
        parking_blend_ramp_s = 0.0  # optional fractional controller flag; initial stand remains1


class Q1TCNCfgPPO(BaseConfig):
    seed = 1
    ablation = "A1"

    class policy:
        command_reference = {}  # opt-in command feedforward, never a body-speed estimate
        wheel_residual = False
        wheel_residual_limit = 0.3
        wheel_residual_yaw_scale = 0.0
        steering_feedforward = {}
        motion_wheel_gain = {}
        parking_policy = False
        parking_blend = False
        freeze_motion_backbone = True  # applies only to optional residual fine-tuning
        parking_settle = {}  # optional history filter using measured gyro/wheel angular speed
        parking_fractional_flag = False
        braking_wheel_extension = {}
        history_length = 16
        channels = 64
        dilations = [1, 2, 4, 8]
        kernel_size = 2
        latent_dim = 16
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        init_noise_std = 0.5
        mean_limit = None
        max_noise_std = 7.38905609893065
        learn_noise = True

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
        max_kl = None  # optional pre-update stop for conservative fine-tuning
        velocity_loss_coef = 1.0
        contact_loss_coef = 1.0
        slip_loss_coef = 1.0
        symmetry_loss_coef = 0.0  # optional left/right policy consistency
        symmetry_batch_size = 4096

    class runner:
        num_steps_per_env = 48
        max_iterations = 3000
        save_interval = 100
        experiment_name = "q1_tcn_reward_v2"


def configure_observation_layout(env_cfg):
    extra = 3 if env_cfg.odometry.enabled else 0
    env_cfg.env.num_observations = 27 + extra
    env_cfg.env.num_privileged_obs = 143 + extra


def validate_config(env_cfg, train_cfg):
    if env_cfg.commands.max_linear_acceleration < 0:
        raise ValueError('max_linear_acceleration must be nonnegative')
    relative=env_cfg.rewards.relative_tracking
    if env_cfg.rewards.reward_version>=7:
        if set(relative)!={'fraction','linear_floor','yaw_floor','zero_yaw_width'} or min(relative.values())<=0:
            raise ValueError('Reward v7 requires positive relative tracking widths')
    elif relative:
        raise ValueError('Relative tracking widths require reward_version>=7')
    wheel_diff=env_cfg.rewards.wheel_diff_tracking
    wheel_weight=env_cfg.rewards.scales.tracking_wheel_diff
    if not math.isfinite(wheel_weight) or wheel_weight<0:
        raise ValueError('Wheel differential tracking weight must be nonnegative')
    if wheel_diff:
        if (set(wheel_diff)!={'track_width','sigma'}
                or any(not math.isfinite(v) or v<=0 for v in wheel_diff.values())):
            raise ValueError('Wheel differential tracking requires positive track_width and sigma')
    elif wheel_weight>0:
        raise ValueError('Enable wheel differential tracking with explicit geometry and sigma')
    configure_observation_layout(env_cfg)
    if not 0 <= env_cfg.commands.sequence_fraction <= 1:
        raise ValueError('sequence_fraction must be in [0, 1]')
    if not 0 < env_cfg.commands.sequence_min_fraction <= 1:
        raise ValueError('sequence_min_fraction must be in (0, 1]')
    if env_cfg.commands.target_stage not in ['linear','turning'] or not 0 <= env_cfg.commands.target_edge_fraction <= 1:
        raise ValueError('Target stage/edge command sampling is invalid')
    extension = train_cfg.policy.braking_wheel_extension
    if extension and max(extension['limit'],extension.get('moving_limit',extension['limit'])) > env_cfg.normalization.clip_actions:
        raise ValueError('Braking wheel extension must fit the executed action limit')
    reference = train_cfg.policy.command_reference
    if reference:
        expected = dict(command_scale=env_cfg.normalization.obs_scales.lin_vel,
                        wheel_radius=env_cfg.asset.wheel_radius,
                        wheel_action_scale=env_cfg.control.vel_action_scale,
                        wheel_velocity_obs_scale=env_cfg.normalization.obs_scales.dof_vel)
        if set(reference)-set(expected)-{'limit','increment_scale','dt','linear_velocity_scale','action_limit','feedforward_gain'}:
            raise ValueError('Unknown command reference setting')
        if reference.get('limit',0)<=0 or any(reference.get(key)!=value for key,value in expected.items()):
            raise ValueError('Command reference must match environment normalization and control scales')
        if env_cfg.odometry.observe_wheel_increments:
            if reference.get('increment_scale')!=env_cfg.odometry.increment_scale:
                raise ValueError('Command reference must shift enabled wheel increment observations')
        if not 0 < reference.get('action_limit', env_cfg.normalization.clip_actions) <= env_cfg.normalization.clip_actions:
            raise ValueError('Command reference action bound must fit the environment clip')
        if reference.get('feedforward_gain',1.) <= 0:
            raise ValueError('Wheel feedforward gain must be positive')
    steering=train_cfg.policy.steering_feedforward
    if steering:
        expected=dict(command_scale=env_cfg.normalization.obs_scales.ang_vel,
                      wheel_radius=env_cfg.asset.wheel_radius,
                      wheel_action_scale=env_cfg.control.vel_action_scale)
        if set(steering)!=set(expected)|{'wheel_y','gain'} or any(steering[k]!=v for k,v in expected.items()):
            raise ValueError('Steering feedforward units must match environment scales')
        if len(steering['wheel_y'])!=2 or not steering['wheel_y'][0]>0>steering['wheel_y'][1] or steering['gain']<=0:
            raise ValueError('Steering requires left-positive/right-negative wheel locations')
    gain=train_cfg.policy.motion_wheel_gain
    if gain:
        if set(gain)!={'start_speed','full_speed','gain','command_scale'}:
            raise ValueError('Motion wheel gain requires speed limits, gain and command scale')
        if not 0<=gain['start_speed']<gain['full_speed'] or gain['gain']<1 or gain['command_scale']!=env_cfg.normalization.obs_scales.lin_vel:
            raise ValueError('Invalid motion wheel output range')
    leg_limit = env_cfg.normalization.leg_clip_actions
    if leg_limit is not None and not 0 < leg_limit <= env_cfg.normalization.clip_actions:
        raise ValueError('Separate leg action bound must fit the global clip')
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
    if env_cfg.odometry.position_scale <= 0:
        raise ValueError("Odometry position_scale must be positive")
    if env_cfg.odometry.parking_observation_delay_s < 0:
        raise ValueError('Parking observation delay must be nonnegative')
    if env_cfg.odometry.parking_blend_ramp_s < 0:
        raise ValueError('Parking blend ramp must be nonnegative')
    if train_cfg.policy.parking_fractional_flag:
        if not env_cfg.odometry.enabled or not train_cfg.policy.parking_policy or not train_cfg.policy.parking_blend or env_cfg.odometry.parking_blend_ramp_s <= 0:
            raise ValueError('Fractional parking flag requires odometry, blended parking branch and positive ramp')
    elif env_cfg.odometry.parking_blend_ramp_s > 0:
        raise ValueError('A fractional parking ramp must be decoded by the policy')
    if env_cfg.odometry.increment_scale <= 0 or (env_cfg.odometry.observe_wheel_increments and not env_cfg.odometry.enabled):
        raise ValueError('Wheel increment observations require enabled encoder odometry and a positive scale')
