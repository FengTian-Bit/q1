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
        permanent_stand_episode_length_s = 0.0  # disabled by default
        obs_history_length = 16

    class terrain(Q1FlatCfg.terrain):
        climb_replay_quarter = False
        climb_replay_max_riser = .165  # default preserves original retention range
        interleave_command_terrain = False
        # Stage 1 slope tiles (Q1T0Terrain). Only read when mesh_type is trimesh;
        # the plane default keeps v46 unchanged.
        terrain_names = ["flat", "slope_up", "slope_down"]
        terrain_labels = [-1, 0, 0]
        slope_range = [0.0, 0.2679]  # rise/run; tan(15 deg) = 0.2679
        difficulty_range = [0.0, 1.0]
        platform_size = 1.0  # small flat spawn platform: most travel is on the slope
        curriculum_move_distance = 1.5  # m from the origin to count as having moved
        spawn_xy_range = 0.0  # metres; opt-in loco_gym-style training reset jitter
        slope_spawn_fraction = 0.0  # share of resets placed on the slope face, not the platform
        slope_spawn_radius = [0.8, 2.5]  # m from the tile centre (platform half-width is 0.5 m)
        seed = None  # set from the env seed when the T0 generator is built
        num_sub_terrains = None  # written by the Terrain generator, saved in snapshots
        selected_type = None  # single-tile evaluation (target_evaluate --stage slope)
        selected_difficulty = 0.0
        stair_height_range = [0.025, 0.20]  # riser range for stairs tiles; stage 2 specs narrow it
        stair_width = 0.25  # nominal tread width; Gym quantizes to grid cells
        stair_spawn_fraction = 0.0  # share of stair-tile resets placed at the low edge
        # r9: on up-climb tiles (inverted pyramid 'stairs_down', class 2) promote only
        # after a world rise >= climb_success_risers * riser; otherwise demote.
        climb_curriculum = False
        climb_success_risers = 1.25
        stair_spawn_radius = [2.6, 3.5]  # m from the tile centre (half tile is 4 m)
        # Rough-slope generator fields (Q1T0Terrain). Only read by rough_slope tiles,
        # which stage 2 adds; the stage 1 mix never referenced them.
        rough_slope_range = [0.1, 0.32]
        rough_height_range = [-0.05, 0.05]
        rough_height_step = 0.005
        rough_downsampled_scale = 0.2
        obstacle_height_range = [0.05, 0.3]
        obstacle_size_range = [1.0, 2.0]
        num_obstacles = 20

    class control(Q1FlatCfg.control):
        decimation = 4  # sim dt .005 s -> policy dt .02 s (50 Hz)
        # Stage 0 per-joint leg scale/limit, order [thigh_L, calf_L, thigh_R, calf_R]
        # in action units. None keeps the single pos_action_scale behaviour.
        # 0.5 x 0.7 = 0.35 rad thigh (unchanged) and 0.9 x 1.35 = 1.215 rad calf.
        leg_action_scales = None
        leg_action_limits = None
        leg_target_margin = 0.05  # rad kept inside the URDF joint limits
        wheel_velocity_feedforward = False  # stage 2: add command vx/r to the wheel target

    class commands(Q1FlatCfg.commands):
        permanent_stand_fraction = 0.0  # whole-episode zero-command environments
        axis_sampling = {}  # optional non-parking straight/pure_yaw/combined proportions
        # r9: share of up-climb-tile (non-stand) resamples given vx in climb_command_vx, yaw 0.
        climb_command_fraction = 0.0
        climb_command_vx = [0.2, 0.6]
        parking_sequence = {}  # optional fraction and speed(m/s) for long-stop trials
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
        clip_constraint_rewards = False  # no caps on penalties; old snapshots explicitly restore True
        reward_version = 2
        support_force = 5.0  # N vertical wheel force counted as support (stage 2 contact reward)
        stair_rise_cap = 0.80  # m cap on the climb potential paid by stair_rise
        nominal_state_swing_exempt = False  # True: no left/right leg-symmetry penalty during a CTBC swing
        ctbc_target_sigma = 0.04  # mean squared joint error in rad^2 (r7 port; default unchanged)
        ctbc_target_unscaled = False  # True: keep the swing-trajectory reward after the teacher feedforward is annealed out
        tracking_sigma = 0.01
        idle_tracking_sigma = 0.01
        yaw_tracking_sigma = 0.04
        relative_tracking = {}  # reward v7: moving-command error widths
        adaptive_tracking_sigma = {}  # optional command-speed smoothstep for Gaussian v2-v5
        turning_linear_sigma_floor = 0.0  # 0 disables the turning-only linear-reward width floor
        parking_drift_stand_only = False  # True: parking_drift applies only to the permanent stand group
        yaw_tracking_mixture = {}  # optional broad/fine yaw feedback; zero yaw retains the original reward
        adaptive_yaw_tracking_sigma = {}  # smooth command-dependent squared yaw-error width
        positive_stand_rewards = {}  # optional zero-command pose/wheel bonuses
        multi_scale_tracking = {}  # reward v8: positive broad/fine Gaussian mixture
        multi_scale_parking = {}  # reward v9: broad acquisition + fine parking costs
        wheel_diff_tracking = {}  # optional track_width(m), sigma((rad/s)^2)
        parking_speed_threshold = 0.02
        parking_yaw_threshold = 0.05
        parking_brake_s = 0.5
        parking_ramp_s = 0.5
        parking_deadband_m = 0.02
        parking_huber_delta_m = 0.0  # disabled; meters after the unchanged deadband
        parking_scale_m = 0.10
        parking_velocity_scale = 0.02
        parking_velocity_deadband = 0.001

        class scales(Q1FlatCfg.rewards.scales):
            alive = 0.0
            nominal_state = 0.0
            stand_still = -0.5
            stand_wheel_still = -1.2
            run_still = -0.05
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
            # Stage 2 teaching signals; zero keeps them out of the reward list.
            ctbc_target = 0.0
            ctbc_clearance = 0.0
            ctbc_contact = 0.0
            # Stage 2 climb signal: pays only the increment of height climbed on
            # stair tiles, so a tall stand earns nothing and a descent pays nothing.
            stair_rise = 0.0

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

    class ctbc:
        """Stage 2 contact-triggered swing teacher. Disabled by default."""
        enabled = False
        contact_threshold = 12.0  # N, horizontal wheel force
        contact_frames = 3
        swing_duration = 0.85  # s, cosine pulse
        min_swing_time = 0.4
        cooldown_s = 0.2
        min_landing_rise = 0.02  # m the wheel must gain before a swing may finish
        clearance_overshoot_allowance = 0.06  # m of extra lift that is free
        guide_lift_ratio = 0.8  # fraction of the riser height to teach
        guide_lift_range = [0.08, 0.22]  # m, clamp on the teaching lift
        feedforward_initial = 1.0
        feedforward_start_iteration = 0
        feedforward_ramp_iterations = 100
        feedforward_hold_iterations = 500
        feedforward_anneal_iterations = 1000
        trigger_enabled = True
        feedforward_enabled = True
        pure_policy = False
        symmetry_mask = True  # suppress the mirror loss while a leg is airborne


class Q1TCNCfgPPO(BaseConfig):
    seed = 1
    ablation = "A1"

    class policy:
        wheel_mean_scale = 1.0  # optional direct wheel mean range; exploration keeps action units
        actor_output_init_gain = None  # optional small random output initialization for fresh PPO
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
        init_noise_std = 1.0
        mean_limit = None
        action_output_scale = None  # Optional policy output units; saved with checkpoints.
        min_noise_std = None
        max_noise_std = None
        learn_noise = True
        # Stage 1: slow TCN over a longer history plus supervised terrain heads.
        # Dilations [4, 8, 16, 32] with kernel 2 see 1 + 60 = 61 frames (1.22 s),
        # so the stored slow history is 61 frames, not 64.
        slow_branch = False
        slow_history_length = 61
        slow_channels = 32
        slow_dilations = [4, 8, 16, 32]
        slow_latent_dim = 16
        slope_head = False
        clearance_head = False
        step_head = False  # blind step-ahead estimator (privileged label, riser rise ahead of each wheel)

    class algorithm:
        learning_rate = 5e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.998
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.0001
        max_grad_norm = 1.0
        desired_kl = 0.01
        max_kl = None  # optional pre-update stop for conservative fine-tuning
        velocity_loss_coef = 1.0
        contact_loss_coef = 1.0
        slip_loss_coef = 1.0
        symmetry_loss_coef = 0.0  # optional left/right policy consistency
        symmetry_batch_size = 4096
        slope_loss_coef = 0.0  # stage 1 supervised terrain heads
        clearance_loss_coef = 0.0
        step_loss_coef = 0.0

    class runner:
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 100
        experiment_name = "q1_tcn_reward_v2"


def configure_observation_layout(env_cfg):
    extra = 3 if env_cfg.odometry.enabled else 0
    env_cfg.env.num_observations = 27 + extra
    env_cfg.env.num_privileged_obs = 143 + extra


def validate_config(env_cfg, train_cfg):
    axes = env_cfg.commands.axis_sampling
    if axes:
        high = axes.get('combined_high', 0.)
        if set(axes)-{'straight', 'pure_yaw', 'combined', 'combined_high', 'combined_high_band'} or not 0 <= high <= 1:
            raise ValueError('Unknown axis_sampling key or combined_high outside [0, 1]')
        axes = {key: axes[key] for key in ['straight', 'pure_yaw', 'combined'] if key in axes}
        if (set(axes) != {'straight', 'pure_yaw', 'combined'}
                or any(not math.isfinite(value) or value <= 0 for value in axes.values())
                or abs(sum(axes.values())-1) > 1e-9
                or env_cfg.commands.stand_probability != 0
                or env_cfg.commands.permanent_stand_fraction > .2
                or env_cfg.commands.sequence_fraction or env_cfg.commands.parking_sequence):
            raise ValueError('Axis sampling requires positive mode proportions summing to1, permanent-only parking<=20%, and no command sequence')
    if not 0<=env_cfg.commands.permanent_stand_fraction<=1:
        raise ValueError('permanent_stand_fraction must be in [0, 1]')
    parking=env_cfg.commands.parking_sequence
    if parking:
        if (set(parking)!={'fraction','speed'}
                or any(not math.isfinite(v) or v<=0 for v in parking.values())
                or env_cfg.commands.permanent_stand_fraction+parking['fraction']>1
                or parking['speed']>min(-env_cfg.commands.ranges.lin_vel_x[0],env_cfg.commands.ranges.lin_vel_x[1])):
            raise ValueError('Parking sequence needs a valid fraction and speed within command bounds')
    wheel_scale=train_cfg.policy.wheel_mean_scale
    if not math.isfinite(wheel_scale) or wheel_scale<=0:
        raise ValueError('Wheel mean scale must be positive and finite')
    if wheel_scale!=1:
        policy=train_cfg.policy
        if (policy.command_reference or policy.motion_wheel_gain or policy.braking_wheel_extension
                or policy.mean_limit is None
                or policy.mean_limit*wheel_scale>env_cfg.normalization.clip_actions):
            raise ValueError('Direct wheel range must fit execution bounds without reference or wheel extensions')
    init_gain=train_cfg.policy.actor_output_init_gain
    if init_gain is not None and (not math.isfinite(init_gain) or not 0<init_gain<=1):
        raise ValueError('Actor output initialization gain must be in (0, 1]')
    if env_cfg.commands.max_linear_acceleration < 0:
        raise ValueError('max_linear_acceleration must be nonnegative')
    positive = env_cfg.rewards.positive_stand_rewards
    if positive:
        if (set(positive) != {'leg_width_rad', 'wheel_width_rad_s', 'zero_command_tolerance'}
                or any(not math.isfinite(v) or v <= 0 for v in positive.values())
                or positive['zero_command_tolerance'] > 1e-5
                or env_cfg.rewards.scales.stand_still <= 0
                or env_cfg.rewards.scales.stand_wheel_still <= 0):
            raise ValueError('Positive stand rewards require positive widths, positive weights and a numerical-zero command tolerance')
    elif env_cfg.rewards.scales.stand_still > 0 or env_cfg.rewards.scales.stand_wheel_still > 0:
        raise ValueError('Positive stand reward weights require explicit positive_stand_rewards settings')
    adaptive = env_cfg.rewards.adaptive_tracking_sigma
    if adaptive:
        if (set(adaptive) != {'full_speed', 'max_sigma'}
                or not 2 <= env_cfg.rewards.reward_version <= 5
                or any(not math.isfinite(v) or v <= 0 for v in adaptive.values())
                or not math.isfinite(env_cfg.rewards.idle_tracking_sigma)
                or env_cfg.rewards.idle_tracking_sigma <= 0
                or not math.isfinite(env_cfg.rewards.parking_speed_threshold)
                or env_cfg.rewards.parking_speed_threshold < 0
                or adaptive['full_speed'] <= env_cfg.rewards.parking_speed_threshold
                or adaptive['max_sigma'] < env_cfg.rewards.idle_tracking_sigma):
            raise ValueError('Adaptive tracking sigma requires Gaussian v2-v5 and positive widths above the parking threshold')
    floor = env_cfg.rewards.turning_linear_sigma_floor
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('Turning linear sigma floor must be finite and nonnegative')
    if floor and (env_cfg.rewards.reward_version != 5 or not adaptive
                  or floor < env_cfg.rewards.idle_tracking_sigma
                  or floor > adaptive['max_sigma']):
        raise ValueError('Turning linear sigma floor requires adaptive v5 and a width within its configured bounds')
    yaw_mixture = env_cfg.rewards.yaw_tracking_mixture
    if yaw_mixture:
        if (set(yaw_mixture) != {'coarse_sigma', 'fine_weight'}
                or env_cfg.rewards.reward_version != 5
                or any(not math.isfinite(v) for v in yaw_mixture.values())
                or not math.isfinite(env_cfg.rewards.yaw_tracking_sigma)
                or env_cfg.rewards.yaw_tracking_sigma <= 0
                or yaw_mixture['coarse_sigma'] <= env_cfg.rewards.yaw_tracking_sigma
                or not 0 < yaw_mixture['fine_weight'] < 1):
            raise ValueError('Yaw mixture requires reward v5, a broader positive squared width and fine_weight in (0, 1)')
    adaptive_yaw = env_cfg.rewards.adaptive_yaw_tracking_sigma
    if adaptive_yaw:
        if (set(adaptive_yaw) != {'full_speed', 'max_sigma'}
                or env_cfg.rewards.reward_version != 5 or yaw_mixture
                or any(not math.isfinite(v) for v in adaptive_yaw.values())
                or not math.isfinite(env_cfg.rewards.yaw_tracking_sigma)
                or env_cfg.rewards.yaw_tracking_sigma <= 0
                or adaptive_yaw['full_speed'] <= env_cfg.rewards.parking_yaw_threshold
                or adaptive_yaw['max_sigma'] < env_cfg.rewards.yaw_tracking_sigma):
            raise ValueError('Adaptive yaw sigma requires reward v5, finite positive widths and no yaw mixture')
    relative=env_cfg.rewards.relative_tracking
    if env_cfg.rewards.reward_version>=7:
        if set(relative)!={'fraction','linear_floor','yaw_floor','zero_yaw_width'} or min(relative.values())<=0:
            raise ValueError('Reward v7 requires positive relative tracking widths')
    elif relative:
        raise ValueError('Relative tracking widths require reward_version>=7')
    multi_scale=env_cfg.rewards.multi_scale_tracking
    if env_cfg.rewards.reward_version>=8:
        required={'linear_coarse_sigma','idle_linear_coarse_sigma','yaw_coarse_sigma','fine_weight'}
        if (set(multi_scale)!=required
                or any(not math.isfinite(v) or v<=0 for v in multi_scale.values())
                or multi_scale['fine_weight']>=1):
            raise ValueError('Reward v8 requires positive finite squared widths and fine_weight in (0, 1)')
    elif multi_scale:
        raise ValueError('Multi-scale tracking requires reward_version>=8')
    parking=env_cfg.rewards.multi_scale_parking
    if env_cfg.rewards.reward_version>=9:
        if (set(parking)!={'position_scale_m','velocity_scale','fine_weight'}
                or any(not math.isfinite(v) or v<=0 for v in parking.values())
                or parking['fine_weight']>=1
                or parking['position_scale_m']<=env_cfg.rewards.parking_scale_m
                or parking['velocity_scale']<=env_cfg.rewards.parking_velocity_scale):
            raise ValueError('Reward v9 requires broader positive parking scales and fine_weight in (0, 1)')
    elif parking:
        raise ValueError('Multi-scale parking requires reward_version>=9')
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
    joint_scales = getattr(env_cfg.control, 'leg_action_scales', None)
    joint_limits = getattr(env_cfg.control, 'leg_action_limits', None)
    if (joint_scales is None) != (joint_limits is None):
        raise ValueError('Per-joint leg scale and limit must be enabled together')
    if joint_scales is not None:
        if len(joint_scales) != 4 or len(joint_limits) != 4:
            raise ValueError('Per-joint leg scale and limit need four entries')
        if any(not math.isfinite(value) or value <= 0 for value in list(joint_scales)+list(joint_limits)):
            raise ValueError('Per-joint leg scale and limit must be positive finite')
        # Limits are in action units; the executed offset is scale x limit (rad).
        if max(joint_limits) > env_cfg.normalization.clip_actions+1e-9:
            raise ValueError('Per-joint leg action bound exceeds the global action clip')
        if leg_limit is not None:
            raise ValueError('Per-joint leg limits replace leg_clip_actions; set only one')
        if not 0 <= env_cfg.control.leg_target_margin < 0.5:
            raise ValueError('Leg target margin must stay below 0.5 rad')
    if train_cfg.ablation not in ABLATIONS:
        raise ValueError("ablation must be one of A0, A1, A2, A3, A4")
    policy = train_cfg.policy
    stored = policy.slow_history_length if policy.slow_branch else policy.history_length
    if env_cfg.env.obs_history_length != stored:
        raise ValueError("env.obs_history_length must equal the slow history (slow branch) "
                         "or the fast history_length")
    if policy.slow_branch and policy.slow_history_length < policy.history_length:
        raise ValueError("Slow history must cover the fast branch window")
    if env_cfg.terrain.mesh_type not in ("plane", "trimesh"):
        raise ValueError("q1_tcn supports a flat plane or T0 trimesh slope tiles")
    if env_cfg.terrain.mesh_type != "plane":
        # Contact/slip labels assume flat ground; only A0 leaves them unsupervised.
        if train_cfg.ablation != "A0":
            raise ValueError("Terrain training requires ablation A0 (flat-only slip/contact labels)")
        if not env_cfg.terrain.measure_heights:
            raise ValueError("Terrain labels need measured heights")
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
    delta = float(getattr(rewards, 'parking_huber_delta_m', 0.))
    if not (delta == 0. or delta == .25) or (delta > 0 and rewards.clip_constraint_rewards):
        raise ValueError('Q1 parking Huber requires delta0(disabled) or0.25m with unclipped rewards')
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
