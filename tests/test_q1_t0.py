"""Exercise Q1's CTBC controller on the ordinary T0 task lifecycle.

These tests execute the real methods with tensor Gym and height-array doubles;
they check integration and observability, not physical climbing performance.
"""

from copy import deepcopy
import unittest

import torch

from test_q1_flat import load_code, make_env
from test_q1_ctbc import NAMESPACE
from test_q1_t0_terrain import build as build_terrain


NAMES = dict(NAMESPACE, deepcopy=deepcopy,
             Q1T0Terrain=lambda cfg, count: build_terrain(cfg)[0])
for filename in ("t0_commands.py", "t0_curriculum.py", "q1_t0_config.py", "q1_t0.py"):
    load_code("wheel_legged_gym/envs/q1/" + filename, NAMES)


def t0_env(count=2, permutation=None):
    env = make_env(count, permutation=permutation, task_class=NAMES["Q1T0CTBC"],
                   cfg_class=NAMES["Q1T0Cfg"])
    env.external_commands = True
    env.command_targets[:, :2] = torch.tensor([0.2, 0.0])
    env.commands[:] = env.command_targets
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    return env


def attach_terrain(env):
    """Attach actual pyramid arrays after the plane-only tensor fixture starts."""
    cfg = env.cfg.terrain
    cfg.mesh_type, cfg.curriculum = "heightfield", True
    cfg.num_rows, cfg.num_cols, cfg.border_size = 3, 2, 0.0
    cfg.terrain_names = ["stairs_down", "stairs_up"]
    cfg.terrain_labels, cfg.terrain_proportions = [2, 3], [0.5, 0.5]
    env.terrain, _ = build_terrain(cfg)
    env.height_samples = torch.as_tensor(env.terrain.height_field_raw)
    env._get_env_origins()
    env.terrain_levels[:] = 1
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env._refresh_terrain_classes()
    env._refresh_climb_targets()
    env.root_states[:, :3] = env.env_origins
    env.root_states[:, 2] += env.cfg.init_state.pos[2]
    env.rigid_body_states[:, env.feet_indices, :2] = env.env_origins[:, None, :2]
    env.rigid_body_states[:, env.feet_indices, 2] = (
        env.env_origins[:, None, 2] + env.cfg.asset.wheel_radius)
    env.base_height[:] = env.cfg.init_state.pos[2]
    env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    env.physical_failure = torch.zeros(env.num_envs, dtype=torch.bool)
    return env


class Q1T0IntegrationTests(unittest.TestCase):
    def test_current_maximum_geometry_and_command_bounds(self):
        cfg = NAMES["Q1T0Cfg"]()
        cfg.terrain.mesh_type = "heightfield"
        cfg.terrain.num_rows, cfg.terrain.border_size = 2, 0.0
        terrain, tools = build_terrain(cfg.terrain)
        torch.testing.assert_close(torch.tensor(terrain.riser_heights[-1, 12:]),
                                   torch.full((8,), .22, dtype=torch.float64))
        torch.testing.assert_close(torch.tensor(terrain.tread_widths[-1, 12:]),
                                   torch.full((8,), .30, dtype=torch.float64))
        slopes = [args["slope"] for kind, _, args in tools.calls if kind == "slope"]
        self.assertAlmostEqual(max(slopes), .48)
        self.assertIn(.32, slopes)
        self.assertIn(-.32, slopes)
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [-1.2, 1.2])
        self.assertEqual(cfg.commands.max_forward_speed, 1.2)

    def test_current_training_distribution_and_stair_columns(self):
        cfg = NAMES["Q1T0Cfg"]().terrain
        cfg.mesh_type, cfg.num_rows, cfg.border_size = "heightfield", 1, 0.0
        self.assertEqual(cfg.terrain_proportions, [.2, .15, .15, .05, .05, .2, .2])
        terrain, _ = build_terrain(cfg)
        self.assertEqual(terrain.terrain_classes[0].tolist(),
                         [-1]*4 + [0]*6 + [1]*2 + [2]*4 + [3]*4)
        self.assertTrue((terrain.step_counts[:, :12] == 0).all())
        self.assertTrue((terrain.step_counts[:, 12:] > 0).all())

    def test_teacher_only_runs_on_stairs_and_clears_nonstair_swing(self):
        env = t0_env(4)
        env.ctbc_step_counts[:] = torch.tensor([0, 0, 3, 3])
        env.set_guidance_mode("on")
        env.commands[:, 0] = env.command_targets[:, 0] = -.3
        env.contact_forces[:, env.feet_indices[0], 0] = 20
        env.guidance.active_leg[:2] = 0
        env.guidance.elapsed[:2] = .3
        for _ in range(8):
            env.step(torch.zeros(4, 6))
        self.assertEqual(env.guidance.active_leg.tolist(), [-1, -1, 0, 0])
        self.assertFalse(env.ctbc_reference[:2].any())
        self.assertTrue(env.ctbc_reference[2:].abs().sum() > 0)
        for reward in (env._reward_ctbc_target, env._reward_ctbc_clearance,
                       env._reward_ctbc_contact):
            self.assertFalse(reward()[:2].any())

    def test_ctbc_inheritance_dimensions_and_blind_actor_contract(self):
        env = t0_env()
        self.assertIsInstance(env, NAMES["Q1CTBC"])
        self.assertEqual(env.cfg.ctbc.stage, "t0")
        env.step(torch.zeros(2, 6))
        self.assertEqual((env.num_actions, env.obs_buf.shape[1],
                          env.privileged_obs_buf.shape[1], env.obs_history.shape[1]),
                         (6, 31, 151, 155))
        self.assertEqual(NAMES["Q1T0CfgPPO"]().policy.num_encoder_obs, 155)
        before = env.compute_proprioception_observations().clone()
        env.ctbc_riser_heights[:] = 0.24
        env.ctbc_guide_lift_heights[:] = 0.16
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = 0.4
        env.contact_forces[:, env.feet_indices, :2] = 100
        env.ctbc_reference[:] = 0.7
        torch.testing.assert_close(env.compute_proprioception_observations(), before)

    def test_high_joint_limits_keep_original_action_to_radian_mapping(self):
        env = t0_env(permutation=[5, 2, 4, 0, 3, 1])
        torch.testing.assert_close(env.ctbc_leg_scales, torch.tensor([0.5, 0.9, 0.5, 0.9]))
        torch.testing.assert_close(env.ctbc_leg_limits, torch.tensor([0.7, 1.35, 0.7, 1.35]))
        torch.testing.assert_close(env.ctbc_action_clip, torch.tensor([1.4, 1.5, 1.0, 1.4, 1.5, 1.0]))
        env.set_guidance_mode("off")
        actions = torch.tensor([[0.4, -0.4, 0.1, -0.2, 0.2, -0.1]]).repeat(2, 1)
        original = actions.clone()
        env.step(actions)
        torch.testing.assert_close(actions, original)
        expected = (env.default_dof_pos[:, env.leg_dof_indices]
                    + actions[:, [0, 1, 3, 4]]*torch.tensor([0.5, 0.9, 0.5, 0.9]))
        torch.testing.assert_close(env.position_target[:, env.leg_dof_indices], expected)
        torch.testing.assert_close(env.obs_buf[:, 21:27], env.ctbc_executed_actions)
        self.assertTrue((env.torques.abs() <= env.torque_limits).all())

    def test_reverse_and_turning_collisions_trigger_but_stationary_contact_does_not(self):
        env = t0_env(3)
        env.ctbc_step_counts[:] = 3
        env.set_guidance_mode("on")
        env.commands[:, :2] = torch.tensor([[-0.3, 0.0], [0.0, -0.4], [0.0, 0.0]])
        env.command_targets[:] = env.commands
        env.contact_forces[:, env.feet_indices[0], 0] = -20.0
        for _ in range(8):
            env.step(torch.zeros(3, 6))
        self.assertEqual(env.guidance.active_leg.tolist(), [0, 0, -1])
        self.assertEqual(env.ctbc_trigger_count.tolist(), [1.0, 1.0, 0.0])
        self.assertTrue((env.actions[:2, 0] > 0).all())
        self.assertEqual(env.actions[2, 0].item(), 0.0)

    def test_no_corridor_terminal_goal_or_world_x_progress_reward(self):
        env = t0_env()
        env.set_guidance_mode("off")
        env.root_states[:, :2] = torch.tensor([[-2.0, 100.0], [10.0, -100.0]])
        env.rigid_body_states[:, env.feet_indices, 0] = 10.0
        env.ctbc_goal_x[:] = -3.0
        env.ctbc_riser_heights[:] = 0
        env.ctbc_step_counts[:] = 0
        env.ctbc_goal_dwell[:] = 0.14
        for _ in range(20):
            _, _, _, done, _, _ = env.step(torch.zeros(2, 6))
            self.assertFalse(done.any())
        self.assertFalse(env.ctbc_success.any())
        self.assertFalse(env.ctbc_out_of_bounds.any())
        self.assertFalse(env.ctbc_progress.any())
        for name in ("ctbc_progress", "ctbc_success", "ctbc_same_x"):
            self.assertEqual(getattr(env.cfg.rewards.scales, name), 0.0)
            self.assertNotIn(name, env.reward_names)

    def test_cross_tile_motion_does_not_use_ctbc_local_corridor_bounds(self):
        env = attach_terrain(t0_env(3))
        env.root_states[:, :2] = torch.tensor([[4.0, 12.0], [12.0, 4.0], [20.0, 12.0]])
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(), [False, False, False])
        self.assertFalse(env.ctbc_out_of_bounds.any())
        self.assertFalse(env.physical_failure.any())
        self.assertFalse(env.time_out_buf.any())

    def test_reset_downgrade_uses_completed_command_and_preserves_neighbor(self):
        env = attach_terrain(t0_env())
        env.metric_steps[:] = 100
        env.commands[:, 0] = torch.tensor([0.5, -0.8])
        env.command_targets[:] = env.commands
        env.root_states[0, 0] += 1.0  # Below .5 m/s * 20 s * .5 = 5 m.
        env.ctbc_trigger_count[:] = torch.tensor([3.0, 7.0])
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = 0.2
        neighbor_amplitudes = env.guidance.amplitudes[1].clone()
        neighbor_root = env.root_states[1].clone()
        env.reset_idx(torch.tensor([0]))
        self.assertEqual(env.terrain_levels.tolist(), [0, 1])
        self.assertEqual(env.fail_ids.tolist(), [0])
        self.assertIsNone(env._t0_reset_commands)
        self.assertEqual(env.commands[1, 0].item(), torch.tensor(-0.8).item())
        torch.testing.assert_close(env.root_states[1], neighbor_root)
        torch.testing.assert_close(env.guidance.amplitudes[1], neighbor_amplitudes)
        self.assertEqual(env.guidance.active_leg.tolist(), [-1, 1])
        self.assertEqual(env.ctbc_trigger_count.tolist(), [0.0, 7.0])
        event = env.extras["q1_t0_episode"]
        self.assertEqual(event["terrain_level"].tolist(), [1])
        self.assertEqual(event["distance_m"].tolist(), [1.0])
        self.assertNotIn("ctbc_episode", env.extras)

    def test_curriculum_refreshes_guide_for_new_row_without_changing_neighbor(self):
        env = attach_terrain(t0_env())
        old_amplitudes = env.guidance.amplitudes.clone()
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = 0.2
        env.commands[:, 0] = 0.2
        env.root_states[0, 0] += 4.01
        env._update_terrain_curriculum(torch.tensor([0]))
        self.assertEqual(env.terrain_levels.tolist(), [2, 1])
        self.assertEqual(env.success_ids.tolist(), [0])
        self.assertEqual(env.env_class.tolist(), [2, 3])
        self.assertAlmostEqual(env.ctbc_riser_heights[0].item(), 0.22)
        self.assertAlmostEqual(env.ctbc_guide_lift_heights[0].item(), 0.16)
        self.assertGreater(env.guidance.amplitudes[0, 0], old_amplitudes[0, 0])
        torch.testing.assert_close(env.guidance.amplitudes[1], old_amplitudes[1])
        self.assertEqual(env.guidance.active_leg.tolist(), [0, 1])
        torch.testing.assert_close(env.guidance.elapsed, torch.full((2,), 0.2))
        torch.testing.assert_close(env.env_origins,
                                   env.terrain_origins[env.terrain_levels, env.terrain_types])

    def test_ten_second_sampling_and_heading_work_with_only_three_command_columns(self):
        env = t0_env()
        # The legacy base allocates an unused fourth column. Remove it to catch
        # accidental calls into its old commands[:, 3] heading implementation.
        for name in ("commands", "command_targets", "applied_commands"):
            setattr(env, name, getattr(env, name)[:, :3].clone())
        env.external_commands = False
        env.commands[:, 0] = -0.5
        env.command_targets[:] = env.commands
        env.command_headings[:] = 0.6
        env.command_ranges["lin_vel_x"] = [0.75, 0.75]
        env.command_ranges["heading"] = [-0.4, -0.4]
        interval = int(10.0/env.dt)
        env.episode_length_buf[:] = torch.tensor([interval-1, interval])
        env._post_physics_step_callback()
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([-0.5, 0.75]))
        torch.testing.assert_close(env.commands[:, 1], torch.tensor([0.3, -0.2]))
        torch.testing.assert_close(env.commands[:, 2], torch.full((2,), env.cfg.commands.ranges.height[0]))
        self.assertEqual(env.commands.shape, (2, 3))
        self.assertEqual(env.command_sample_step[1].item(), interval)
        env.episode_length_buf[:] = interval+1
        env.step(torch.zeros(2, 6))
        self.assertEqual(env.obs_buf.shape, (2, 31))

    def test_physical_failure_overrides_timeout_and_survives_reset_event(self):
        env = t0_env(4)
        env.metric_steps[:] = 100
        env.contact_forces[0, env.termination_contact_indices[0], 2] = 10.0
        env.projected_gravity[1, 2] = 0.0
        env.base_height[2] = 0.05
        env.episode_length_buf[:] = env.max_episode_length
        env.check_termination()
        self.assertEqual(env.physical_failure.tolist(), [True, True, True, False])
        self.assertEqual(env.time_out_buf.tolist(), [False, False, False, True])
        self.assertTrue(env.reset_buf.all())
        env.reset_idx(torch.arange(4))
        event = env.extras["q1_t0_episode"]
        self.assertEqual(event["physical_failure"].tolist(), [True, True, True, False])
        self.assertEqual(event["timeout"].tolist(), [False, False, False, True])
        self.assertNotIn("success", event)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
