"""High-stair instruction and actuator-target checks; no contact simulation.

The real URDF property callback is used here: its 0.97 soft limits are tighter
than the raw limits installed by the generic tensor fixture.
"""
import math
import unittest

import numpy as np
import torch

from test_q1_flat import NS, load_code, make_env
from test_q1_ctbc import NAMESPACE
from test_q1_ctbc_terrain import HeightfieldTools


terrain_namespace = dict(NS, terrain_utils=HeightfieldTools())
load_code("wheel_legged_gym/utils/terrain.py", terrain_namespace)
load_code("wheel_legged_gym/envs/q1_ctbc/terrain.py", terrain_namespace)


def high_config():
    cfg = NAMESPACE["Q1CTBCCfg"]()
    return NAMESPACE["configure_ctbc_stage"](cfg, "stairs20")


def fk_z(q):
    """Hip-to-wheel-centre vertical coordinate for Q1's two 0.2 m links."""
    return -0.2 * torch.cos(q[..., 0]) - 0.2 * torch.cos(q.sum(dim=-1))


def make_high_env(count=12):
    env = make_env(count, task_class=NAMESPACE["Q1CTBC"], cfg_class=high_config)
    # Run the production URDF callback rather than checking against the tensor
    # fixture's raw limits. This also keeps importer DOF ordering intact.
    props = np.zeros(env.num_dof, dtype=[(name, np.float32)
                                       for name in ("lower", "upper", "velocity", "effort")])
    props["lower"] = env.dof_pos_limits[:, 0].numpy()
    props["upper"] = env.dof_pos_limits[:, 1].numpy()
    props["velocity"] = env.dof_vel_limits.numpy()
    props["effort"] = env.torque_limits.numpy()
    env._process_dof_props(props, 0)
    env.cfg.terrain.mesh_type = "heightfield"
    terrain = terrain_namespace["CTBCTerrain"](env.cfg.terrain, count)
    env.terrain = terrain
    env.custom_origins = True
    env.terrain_levels = torch.arange(count) % env.cfg.terrain.num_rows
    env.terrain_types = torch.ones(count, dtype=torch.long)
    env.max_terrain_level = env.cfg.terrain.num_rows
    env.terrain_origins = torch.tensor(terrain.env_origins, dtype=torch.float32)
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.terrain_class_map = torch.tensor(terrain.terrain_classes, dtype=torch.long)
    env.height_samples = torch.tensor(terrain.heightsamples)
    env.ctbc_terrain_maps = {
        name: torch.tensor(getattr(terrain, name))
        for name in ("riser_heights", "tread_widths", "step_counts", "first_riser_x", "goal_x")}
    env._refresh_terrain_classes()
    env._refresh_climb_targets()
    env.root_states[:, :3] = env.env_origins + torch.tensor([0.0, 0.0, 0.402])
    env.rigid_body_states[:, env.feet_indices, :3] = env.env_origins[:, None, :]
    env.rigid_body_states[:, env.feet_indices, 2] = env.cfg.asset.wheel_radius
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    env.external_commands = True
    env.commands[:, 0] = env.command_targets[:, 0] = 0.2
    return env


class Q1CTBCHighTests(unittest.TestCase):
    def test_high_stage_has_twelve_quantized_rows_and_a_true_twenty_two_cm_top_row(self):
        cfg = high_config()
        cfg.terrain.mesh_type = "heightfield"
        terrain = terrain_namespace["CTBCTerrain"](cfg.terrain, 32)
        np.testing.assert_allclose(terrain.riser_heights[:, 1],
                                   [.04, .055, .07, .085, .105, .12, .135, .15, .17, .185, .20, .22])
        np.testing.assert_allclose(terrain.tread_widths[:, 1], .50)
        self.assertTrue((terrain.step_counts[:, 1] == 3).all())
        start_x = terrain.border + 11 * terrain.length_per_env_pixels
        start_y = terrain.border + terrain.width_per_env_pixels
        heights = terrain.height_field_raw[
            start_x:start_x + terrain.length_per_env_pixels,
            start_y:start_y + terrain.width_per_env_pixels][:, 60]
        edges = np.flatnonzero(np.diff(heights.astype(np.int32))) + 1
        np.testing.assert_array_equal(edges, [68, 88, 108])
        np.testing.assert_allclose((heights[edges] - heights[edges - 1]) * .005, .22)
        self.assertAlmostEqual(heights[-1] * .005, .66)
        self.assertAlmostEqual(terrain.goal_x[11, 1], 66 + 3.0)
        self.assertLess(terrain.goal_x[11, 1] + .3, 72.0)
        np.testing.assert_allclose(terrain.env_origins[:, :, 2], 0)

    def test_high_stage_back_to_each_existing_stage_restores_all_changed_controls(self):
        configure = NAMESPACE["configure_ctbc_stage"]
        for stage in ("single", "stairs", "robust"):
            with self.subTest(stage=stage):
                cfg = configure(high_config(), stage)
                expected = configure(NAMESPACE["Q1CTBCCfg"](), stage)
                for section in ("control", "ctbc", "normalization", "rewards", "commands", "terrain"):
                    self.assertEqual(NS["class_to_dict"](getattr(cfg, section)),
                                     NS["class_to_dict"](getattr(expected, section)))
                self.assertEqual(cfg.control.leg_action_scales, [.5, .9, .5, .9])
                self.assertEqual(cfg.normalization.clip_actions, 1.0)
                self.assertAlmostEqual(cfg.ctbc.swing_duration, .6)
                self.assertAlmostEqual(cfg.ctbc.min_swing_time, .3)
                self.assertAlmostEqual(cfg.rewards.scales.ctbc_target, 1.0)

    def test_existing_normalized_actions_preserve_physical_mapping_in_high_stage(self):
        env = make_high_env(2)
        env.set_guidance_mode("off")
        actions = torch.tensor([[1., -1., .4, -.7, .8, -.2],
                                [-1., 1., -.3, .6, -.9, .5]])
        before = actions.clone()
        env.step(actions)
        expected = (env.default_dof_pos[:, env.leg_dof_indices]
                    + actions[:, [0, 1, 3, 4]] * torch.tensor([.5, .9, .5, .9]))
        torch.testing.assert_close(env.position_target[:, env.leg_dof_indices], expected)
        torch.testing.assert_close(actions, before)
        self.assertEqual(env.obs_buf.shape[1], 31)
        self.assertEqual(env.obs_history.shape[1], 155)

    def test_guidance_scales_per_row_and_seventeen_point_six_cm_survives_clipping(self):
        env = make_high_env()
        torch.testing.assert_close(env.ctbc_guide_lift_heights,
                                   torch.tensor([.08, .08, .08, .08, .084, .096,
                                                 .108, .12, .136, .148, .16, .176]),
                                   check_dtype=False)
        env.set_guidance_mode("on")
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration / 2 - env.dt
        env.step(torch.zeros(env.num_envs, 6))
        default = env.default_dof_pos[:, env.leg_dof_indices[:2]]
        target = env.position_target[:, env.leg_dof_indices[:2]]
        torch.testing.assert_close(fk_z(target) - fk_z(default),
                                   env.ctbc_guide_lift_heights, atol=2e-6, rtol=0,
                                   check_dtype=False)
        self.assertAlmostEqual(target[-1, 0].item(), 1.1752248307, places=5)
        self.assertAlmostEqual(target[-1, 1].item(), -2.3504496613, places=5)
        self.assertGreater(env.actions[-1, 0].item(), 1.0)
        self.assertLess(env.actions[-1, 1].item(), -1.0)
        torch.testing.assert_close(env.position_target[:, env.leg_dof_indices[2:]],
                                   env.default_dof_pos[:, env.leg_dof_indices[2:]])
        limits = env.dof_pos_limits[env.leg_dof_indices[:2]]
        self.assertTrue((target >= limits[:, 0] + .05 - 1e-6).all())
        self.assertTrue((target <= limits[:, 1] - .05 + 1e-6).all())

    def test_pure_policy_can_request_twenty_cm_with_real_soft_limits_and_no_teacher(self):
        env = make_high_env(1)
        env.set_guidance_mode("off")
        # IK from the existing 0.6/-1.2 pose, keeping wheel X under its hip.
        depth = .4 * math.cos(.6)
        hip = math.acos((depth - .20) / .4)
        offsets = torch.tensor([hip - .6, -2 * hip + 1.2])
        actions = torch.zeros(1, 6)
        actions[0, :2] = offsets / torch.tensor([.5, .9])
        self.assertTrue((actions.abs() <= env.ctbc_action_clip + 1e-6).all())
        env.step(actions)
        q = env.position_target[:, env.leg_dof_indices[:2]]
        base = env.default_dof_pos[:, env.leg_dof_indices[:2]]
        self.assertAlmostEqual((fk_z(q) - fk_z(base)).item(), .20, places=6)
        # These are software limits after applying URDF soft_dof_pos_limit=.97.
        self.assertAlmostEqual(env.dof_pos_limits[env.leg_dof_indices[1], 0].item(), -2.561, places=5)
        self.assertAlmostEqual(q[0, 1].item(), -2.4788593943, places=5)
        self.assertGreater(q[0, 1].item(), -2.511)
        self.assertEqual(env.ctbc_feedforward_weight, 0.0)
        self.assertFalse(env.guidance.swing_mask.any())

    def test_reference_velocity_stays_below_urdf_and_acceleration_below_old_case(self):
        env = make_high_env(1)
        theta = math.acos((.4 * math.cos(.6) - .176) / .4)
        amplitude = theta - .6
        duration = env.cfg.ctbc.swing_duration
        velocity = torch.tensor([amplitude, 2 * amplitude]) * math.pi / duration
        self.assertTrue((velocity < env.dof_vel_limits[env.leg_dof_indices[:2]]).all())
        self.assertAlmostEqual(velocity[1].item(), 4.2520520051, places=5)
        old_knee = .5905 * 2 * math.pi**2 / .6**2
        new_knee = 2 * amplitude * 2 * math.pi**2 / duration**2
        self.assertLess(new_knee, old_knee)
        self.assertAlmostEqual(env.cfg.ctbc.min_swing_time, .4)

    def test_clearance_accepts_twenty_two_with_new_reference_but_penalizes_excess(self):
        env = make_high_env()
        env.set_guidance_mode("on")
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration / 2
        env.guidance.start_height[:] = 0.0
        env.ctbc_reference[:] = env.guidance.reference_offsets()
        rewards = []
        for height in (.176, .22, .236, .26):
            env.rigid_body_states[:, env.feet_indices[0], 2] = env.cfg.asset.wheel_radius + height
            rewards.append(env._reward_ctbc_clearance()[-1].item())
        self.assertGreaterEqual(rewards[1] + 1e-6, rewards[0])
        self.assertGreaterEqual(rewards[2] + 1e-6, rewards[1])
        self.assertLess(rewards[3], rewards[2] - .05)

    def test_joint_target_reward_releases_as_feedforward_anneals_to_zero(self):
        env = make_high_env(1)
        env.set_guidance_mode("annealed")
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration / 2
        env.ctbc_reference[:] = env.guidance.reference_offsets()
        env.dof_pos[:, env.leg_dof_indices] = (
            env.default_dof_pos[:, env.leg_dof_indices] + env.ctbc_reference)
        hold, anneal = (env.cfg.ctbc.feedforward_hold_iterations,
                        env.cfg.ctbc.feedforward_anneal_iterations)
        values = []
        for iteration in (hold, hold + anneal // 2, hold + anneal):
            env.set_training_iteration(iteration)
            values.append(env._reward_ctbc_target().item())
        self.assertGreater(values[0], 0)
        self.assertAlmostEqual(values[1], values[0] / 2, places=6)
        self.assertEqual(values[2], 0.0)

    def test_high_guidance_enters_gradually_before_hold_then_anneals(self):
        env = make_high_env(1)
        env.set_guidance_mode("annealed")
        cfg = env.cfg.ctbc
        self.assertEqual(cfg.feedforward_ramp_iterations, 1000)
        values = []
        for iteration in (0, 500, 1000, cfg.feedforward_hold_iterations,
                          cfg.feedforward_hold_iterations + cfg.feedforward_anneal_iterations):
            env.set_training_iteration(iteration)
            values.append(env.ctbc_feedforward_weight)
        np.testing.assert_allclose(values, [0, .5, 1, 1, 0])

    def test_guide_rejects_offset_overflow_and_tighter_software_joint_margin(self):
        env = make_high_env(1)
        env.cfg.ctbc.guide_lift_height_range = [.24, .24]
        with self.assertRaisesRegex(ValueError, "configured leg offset limits"):
            env._refresh_climb_targets()

        # A 22 cm symmetric IK pose fits the raw -2.6 rad URDF knee limit,
        # but violates the production software margin -2.511 rad. Expand only
        # the offset bound in this diagnostic to exercise the tighter check.
        env.cfg.ctbc.guide_lift_height_range = [.22, .22]
        env.cfg.control.leg_action_limits = [2., 2., 2., 2.]
        env.ctbc_leg_limits[:] = 2.0
        theta = math.acos((.4 * math.cos(.6) - .22) / .4)
        self.assertGreater(-2 * theta, -2.6)
        self.assertLess(-2 * theta, -2.511)
        with self.assertRaisesRegex(ValueError, "joint target margins"):
            env._refresh_climb_targets()

    def test_terrain_target_refresh_changes_only_the_changed_row_instruction(self):
        env = make_high_env(2)
        env.terrain_levels[:] = torch.tensor([0, 11])
        env._refresh_climb_targets()
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration / 2
        before = env.guidance.reference_offsets().clone()
        active, elapsed = env.guidance.active_leg.clone(), env.guidance.elapsed.clone()
        env.terrain_levels[0] = 9
        env._refresh_climb_targets()
        after = env.guidance.reference_offsets()
        torch.testing.assert_close(after[1], before[1])
        torch.testing.assert_close(env.guidance.active_leg, active)
        torch.testing.assert_close(env.guidance.elapsed, elapsed)
        self.assertAlmostEqual(env.ctbc_guide_lift_heights[0].item(), .148, places=6)
        self.assertAlmostEqual(env.ctbc_guide_lift_heights[1].item(), .176, places=6)
        q0 = env.default_dof_pos[0, env.leg_dof_indices[:2]]
        self.assertAlmostEqual((fk_z(q0 + after[0, :2]) - fk_z(q0)).item(), .148, places=6)

    def test_three_twenty_two_cm_steps_do_not_create_absolute_height_termination(self):
        env = make_high_env()
        top = env.ctbc_riser_heights * env.ctbc_step_counts
        env.root_states[:, 0] = env.ctbc_goal_x + .1
        env.root_states[:, 2] = top + .397
        env.rigid_body_states[:, env.feet_indices, 0] = env.ctbc_goal_x[:, None].float() + .1
        env.rigid_body_states[:, env.feet_indices, 2] = top[:, None].float() + env.cfg.asset.wheel_radius
        for _ in range(15):
            env._post_physics_step_callback()
        env.check_termination()
        self.assertFalse(env.physical_failure.any())
        self.assertTrue(env.ctbc_success.all())
        self.assertAlmostEqual(top[-1].item(), .66, places=6)
        self.assertAlmostEqual(env.base_height[-1].item(), .397, places=5)


if __name__ == "__main__":
    unittest.main()
