"""Full forward/reverse command range and soft leg symmetry on mixed terrain."""
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import torch

from test_q1 import improved
from test_q1_flat import NS, load_code
from test_q1_terrain import rough_env


class Q1SpeedTests(unittest.TestCase):
    def test_stage_sampling_reaches_both_endpoints_on_flat_and_rough_ground(self):
        for stage in ("low_speed", "turning"):
            with self.subTest(stage=stage):
                env = rough_env(4096)
                NS["configure_q1_stage"](env.cfg, stage)
                env.env_class[:2048] = -1
                env.needs_command_sample[:] = True
                env._resample_commands(torch.arange(4096))
                for speed in (env.command_targets[:2048, 0], env.command_targets[2048:, 0]):
                    self.assertTrue((speed.abs() <= 0.8).all())
                    self.assertTrue((speed > 0.799).any())
                    self.assertTrue((speed < -0.799).any())
        NS["configure_q1_stage"](env.cfg, "balance")
        env.needs_command_sample[:] = True
        env._resample_commands(torch.arange(4096))
        self.assertEqual(env.command_targets[:, :2].abs().sum().item(), 0)
        self.assertEqual(NS["Q1FlatCfg"]().commands.ranges.lin_vel_x, [-0.3, 0.3])

    def test_manual_limit_and_slew_reach_full_speed_in_both_directions(self):
        env = improved(2)
        env.set_command_target(torch.tensor([2.0, -2.0]))
        env.commands[:, :2] = 0
        torch.testing.assert_close(env.command_targets[:, 0], torch.tensor([0.8, -0.8]))
        for _ in range(200):
            env._post_physics_step_callback()
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([0.8, -0.8]), atol=2e-6, rtol=0)
        self.assertAlmostEqual(env.cfg.commands.acceleration_limit, 0.4)
        obs = env.compute_proprioception_observations()
        self.assertEqual(obs.shape[1], 31)
        torch.testing.assert_close(obs[:, 6], env.commands[:, 0]*env.commands_scale[0])

    def test_full_speed_and_yaw_feedforward_fit_existing_urdf_wheel_limits(self):
        env = improved(4)
        env.commands[:, :2] = torch.tensor([[0.8, 0.6], [0.8, -0.6], [-0.8, 0.6], [-0.8, -0.6]])
        torques = env._compute_torques(torch.zeros(4, 6))
        expected = (env.commands[:, :1]-env.commands[:, 1:2]*env.wheel_lateral_offsets)/env.cfg.asset.wheel_radius
        torch.testing.assert_close(env.velocity_target[:, env.wheel_dof_indices], expected)
        self.assertLess(expected.abs().max().item(), 12.0)
        self.assertTrue((expected.abs() < env.dof_vel_limits[env.wheel_dof_indices]).all())
        self.assertTrue((torques.abs() <= env.torque_limits).all())

    def test_restored_symmetry_on_rough_ground_keeps_tolerance_and_turn_relaxation(self):
        env = rough_env(4)
        env.env_class = torch.tensor([-1, 2, 3, 4])
        env.command_acceleration[:] = env.yaw_acceleration[:] = 0
        env.commands[:, 1] = 0
        env.dof_pos[:, env.leg_dof_indices[0]] += 0.04
        self.assertEqual(env._reward_leg_symmetry().sum().item(), 0)
        env.dof_pos[:, env.leg_dof_indices[0]] += 0.3
        straight = env._reward_leg_symmetry().clone()
        self.assertTrue((straight > 0).all())
        self.assertLess(env.reward_scales["leg_symmetry"], 0)
        env.commands[:, 1] = 0.6
        torch.testing.assert_close(env._reward_leg_symmetry(), 0.5*straight)
        env.cfg.rewards.rough_leg_symmetry_weight = 0.0
        self.assertEqual(env._reward_leg_symmetry()[1:].sum().item(), 0)

    def test_speed_cost_distinguishes_large_errors_and_recording_accepts_full_range(self):
        env = improved(5)
        env.applied_commands[:, 0] = 0.8
        env.base_lin_vel[:, 0] = torch.tensor([0.8, 0.6, 0.3, 0.0, -0.8])
        cost = env._reward_speed_error()
        self.assertEqual(cost[0].item(), 0)
        self.assertTrue((cost[1:] > cost[:-1]).all())
        namespace = dict(NS, Path=Path, shutil=shutil, subprocess=subprocess, sys=sys)
        load_code("wheel_legged_gym/rsl_rl/utils/video.py", namespace)
        schedule = [(1, 0.8, 0.6), (1, -0.8, -0.6)]
        self.assertEqual(namespace["video_settings"]({"command_schedule": schedule})["command_schedule"], schedule)
        with self.assertRaises(ValueError):
            namespace["video_settings"]({"command_schedule": [(1, 0.81, 0)]})
        configured = NS["Q1CfgPPO"]().runner.periodic_video.command_schedule
        self.assertEqual(max(abs(entry[1]) for entry in configured), 0.8)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
