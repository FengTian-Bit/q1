"""Contact-guided climbing curriculum for Q1; all distances are in metres."""

from wheel_legged_gym.envs.q1.q1_config import Q1Cfg, Q1CfgPPO


class Q1CTBCCfg(Q1Cfg):
    class terrain(Q1Cfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        horizontal_scale = 0.025
        vertical_scale = 0.005
        border_size = 2.0
        terrain_length = 6.0
        terrain_width = 3.0
        num_rows = 6
        num_cols = 5
        max_init_terrain_level = 0
        terrain_names = ["flat", "stairs_up"]
        terrain_labels = [-1, 3]
        terrain_proportions = [0.2, 0.8]
        riser_height_range = [0.025, 0.08]
        tread_width_range = [0.5, 0.5]
        num_steps = 1
        spawn_x = 1.0
        first_step_x = 1.7
        goal_margin = 0.3
        flat_goal_distance = 1.8
        selected = False
        selected_height = 0.05
        selected_tread = 0.5
        selected_steps = 1
        selected_flat = False

    class commands(Q1Cfg.commands):
        resampling_time = 20.0
        max_forward_speed = 0.3
        minimum_motion_speed = 0.15
        stand_probability = 0.0
        straight_probability = 1.0
        in_place_probability = 0.0
        endpoint_probability = 0.0

        class ranges(Q1Cfg.commands.ranges):
            lin_vel_x = [0.15, 0.3]
            ang_vel_yaw = [0.0, 0.0]

    class control(Q1Cfg.control):
        # Policy leg order: L thigh, L calf, R thigh, R calf.
        leg_action_scales = [0.5, 0.9, 0.5, 0.9]

    class ctbc:
        stage = "single"
        contact_threshold = 12.0
        contact_frames = 3
        swing_duration = 0.6
        min_swing_time = 0.3
        cooldown_s = 0.2
        # Peak joint offsets, with signs verified against Q1's +Y joint axes.
        hip_amplitude = 0.2952
        knee_amplitude = -0.5905
        feedforward_initial = 1.0
        feedforward_hold_iterations = 3000
        feedforward_anneal_iterations = 12000
        trigger_enabled = True
        feedforward_enabled = True
        pure_policy = False
        goal_dwell_s = 0.15
        curriculum_window = 5
        curriculum_promote_rate = 0.8
        curriculum_demote_rate = 0.3

    class rewards(Q1Cfg.rewards):
        tracking_sigma = 0.04
        rough_leg_symmetry_weight = 1.0

        class scales(Q1Cfg.rewards.scales):
            leg_symmetry = -0.10
            speed_error = -0.1
            ctbc_target = 1.0
            ctbc_clearance = 1.0
            ctbc_contact = 1.0
            ctbc_progress = 0.5
            ctbc_success = 5.0
            ctbc_same_x = 0.0


class Q1CTBCCfgPPO(Q1CfgPPO):
    class runner(Q1CfgPPO.runner):
        experiment_name = "q1_ctbc"
        run_name = "single"
        swanlab_tags = ["q1", "ctbc"]

        class periodic_video(Q1CfgPPO.runner.periodic_video):
            enabled = False


def configure_ctbc_stage(cfg, stage):
    """Apply a complete stage, so switching back also removes robust settings."""
    if stage not in ("single", "stairs", "robust"):
        raise ValueError("Unknown CTBC stage: {}".format(stage))
    cfg.ctbc.stage = stage
    cfg.commands.ranges.lin_vel_x = [0.15, 0.3]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.stand_probability = 0.0
    cfg.commands.straight_probability = 1.0
    cfg.commands.in_place_probability = 0.0
    cfg.terrain.riser_height_range = [0.025, 0.08] if stage == "single" else [0.04, 0.12]
    cfg.terrain.tread_width_range = [0.5, 0.5] if stage == "single" else [0.5, 0.35]
    cfg.terrain.num_steps = 1 if stage == "single" else 3
    # These are conservative Q1 starting ranges, not claimed paper parameters.
    robust = stage == "robust"
    for name in ("randomize_friction", "randomize_base_mass", "randomize_inertia",
                 "randomize_base_com", "randomize_Kp", "randomize_Kd",
                 "randomize_motor_torque", "randomize_default_dof_pos",
                 "randomize_action_delay", "push_robots"):
        setattr(cfg.domain_rand, name, robust)
    # The inherited inertia randomizer also rescales link masses, after the
    # critic mass cache is recorded. Keep mass variation explicit for this task.
    cfg.domain_rand.randomize_inertia = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.friction_range = [0.6, 1.2]
    cfg.domain_rand.added_mass_range = [-0.2, 0.4]
    cfg.domain_rand.randomize_inertia_range = [0.9, 1.1]
    cfg.domain_rand.rand_com_vec = [0.01, 0.01, 0.01]
    cfg.domain_rand.randomize_Kp_range = [0.9, 1.1]
    cfg.domain_rand.randomize_Kd_range = [0.9, 1.1]
    cfg.domain_rand.randomize_motor_torque_range = [0.9, 1.1]
    cfg.domain_rand.randomize_default_dof_pos_range = [-0.02, 0.02]
    cfg.domain_rand.delay_ms_range = [0, 10] if robust else [0, 0]
    cfg.domain_rand.push_interval_s = 8.0
    cfg.domain_rand.max_push_vel_xy = 0.3
    cfg.noise.add_noise = robust
    return cfg
