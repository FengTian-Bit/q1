"""Validate T0 command distribution and heading control with Q1's layout."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q1_t0_commands", ROOT / "wheel_legged_gym/envs/q1/t0_commands.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_env(count=4, heading=True):
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            heading_command=heading,
            resampling_time=10.0,
            ranges=SimpleNamespace(height=[0.397, 0.397]),
        ),
        terrain=SimpleNamespace(highplatform_terrain_type=5),
    )
    return SimpleNamespace(
        cfg=cfg,
        dt=0.01,
        device="cpu",
        external_commands=False,
        commands=torch.zeros(count, 3),
        command_targets=torch.zeros(count, 3),
        command_ranges={
            "lin_vel_x": [-1.8, 1.8],
            "ang_vel_yaw": [-1.0, 1.0],
            "heading": [-3.14, 3.14],
        },
        base_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(count, 1),
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        needs_command_sample=torch.ones(count, dtype=torch.bool),
        command_sample_step=torch.zeros(count, dtype=torch.long),
    )


class T0CommandTests(unittest.TestCase):
    def test_sampling_is_uniform_without_q1_task_probabilities(self):
        env = make_env()
        draws = [torch.tensor([0.0, 0.25, 0.75, 1.0]), torch.tensor([0.0, 0.25, 0.75, 1.0])]
        with patch.object(MODULE.torch, "rand", side_effect=draws) as draw:
            MODULE.resample_commands(env, torch.arange(4))
        self.assertEqual(draw.call_count, 2)
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([-1.8, -0.9, 0.9, 1.8]))
        torch.testing.assert_close(env.command_headings, torch.tensor([-3.14, -1.57, 1.57, 3.14]))
        torch.testing.assert_close(env.commands[:, 2], torch.full((4,), 0.397))
        torch.testing.assert_close(env.command_targets, env.commands)
        self.assertFalse(env.needs_command_sample.any())

    def test_translation_deadband_excludes_equal_threshold_and_keeps_both_signs(self):
        env = make_env(6, heading=False)
        env.command_ranges["lin_vel_x"] = [-1.0, 1.0]
        # Exactly representable 0.2 bounds guarantee an exact boundary check.
        env.command_ranges["lin_vel_x"] = torch.tensor([
            [-0.21, -0.21], [-0.2, -0.2], [0.0, 0.0],
            [0.2, 0.2], [0.21, 0.21], [0.9, 0.9],
        ])
        MODULE.resample_commands(env, torch.arange(6))
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([-0.21, 0.0, 0.0, 0.0, 0.21, 0.9]))

    def test_direct_yaw_mode_uses_q1_yaw_slot_and_preserves_height(self):
        env = make_env(2, heading=False)
        env.command_ranges["ang_vel_yaw"] = torch.tensor([[-0.75, -0.75], [0.6, 0.6]])
        MODULE.resample_commands(env, torch.arange(2))
        MODULE.update_heading_command(env)
        torch.testing.assert_close(env.commands[:, 1], torch.tensor([-0.75, 0.6]))
        torch.testing.assert_close(env.commands[:, 2], torch.full((2,), 0.397))
        self.assertFalse(hasattr(env, "command_headings"))

    def test_heading_gain_saturation_and_pi_wrap_match_t0(self):
        env = make_env(5)
        env.command_headings = torch.tensor([0.6, 3.0, -3.0, math.pi, -math.pi])
        MODULE.update_heading_command(env)
        torch.testing.assert_close(env.commands[:, 1], torch.tensor([0.3, 1.0, -1.0, 1.0, 1.0]))
        torch.testing.assert_close(env.command_targets[:, 1], env.commands[:, 1])

    def test_heading_tracks_current_robot_heading_each_step_without_resampling(self):
        env = make_env(1)
        env.command_headings = torch.tensor([-3.0])
        env.episode_length_buf[:] = 1
        heading = 3.0
        env.base_quat[0] = torch.tensor([0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2)])
        with patch.object(MODULE, "resample_commands") as sample:
            MODULE.compute_commands(env)
        # compute_commands may dispatch an empty selection, as T0's manager
        # does; no environment is resampled between its ten-second periods.
        self.assertEqual(sample.call_args.args[1].numel(), 0)
        self.assertAlmostEqual(env.commands[0, 1].item(), (2 * math.pi - 6) / 2, places=6)
        env.base_quat[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        MODULE.update_heading_command(env)
        self.assertEqual(env.commands[0, 1].item(), -1.0)

    def test_ten_second_sampling_only_updates_due_environments(self):
        env = make_env(3)
        env.episode_length_buf[:] = torch.tensor([999, 1000, 2000])
        env.command_headings = torch.zeros(3)
        with patch.object(MODULE, "resample_commands") as sample:
            MODULE.compute_commands(env)
        torch.testing.assert_close(sample.call_args.args[1], torch.tensor([1, 2]))

    def test_per_environment_ranges_and_subset_sampling(self):
        env = make_env(3)
        env.command_ranges["lin_vel_x"] = torch.tensor([[-0.6, -0.6], [0.8, 0.8], [1.2, 1.2]])
        env.command_ranges["heading"] = torch.tensor([[-1.0, -1.0], [0.5, 0.5], [1.0, 1.0]])
        env.commands[0] = torch.tensor([9.0, 8.0, 7.0])
        env.episode_length_buf[:] = torch.tensor([2, 3, 4])
        MODULE.resample_commands(env, torch.tensor([2, 1]))
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([9.0, 0.8, 1.2]))
        torch.testing.assert_close(env.command_headings[1:], torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(env.command_sample_step, torch.tensor([0, 3, 4]))
        self.assertTrue(env.needs_command_sample[0])

    def test_optional_terrain_speed_scaling_follows_deadband(self):
        env = make_env(3, heading=False)
        env.env_class = torch.tensor([-1, -1, 3])
        env.cfg.commands.terrain_speed_scale = SimpleNamespace(terrain_types=[-1], scale=2.0)
        env.command_ranges["lin_vel_x"] = torch.tensor([[0.1, 0.1], [0.3, 0.3], [0.3, 0.3]])
        MODULE.resample_commands(env, torch.arange(3))
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([0.0, 0.6, 0.3]))

    def test_optional_highplatform_override_retains_source_order(self):
        env = make_env(2)
        env.env_class = torch.tensor([5, -1])
        env.cfg.commands.highplatform = SimpleNamespace(lin_vel_x=[0.1, 0.1], heading=[0.05, 0.05])
        env.command_ranges["lin_vel_x"] = [0.5, 0.5]
        MODULE.resample_commands(env, torch.arange(2))
        # T0 overrides after deadband: a platform-specific 0.1 survives.
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([0.1, 0.5]))
        self.assertAlmostEqual(env.command_headings[0].item(), 0.05)

    def test_external_commands_bypass_sampling_and_heading_processing(self):
        env = make_env(1)
        env.external_commands = True
        env.commands[0] = torch.tensor([0.5, -0.3, 0.397])
        before = env.commands.clone()
        MODULE.resample_commands(env, torch.tensor([0]))
        MODULE.update_heading_command(env)
        torch.testing.assert_close(env.commands, before)
        env.command_targets[0] = torch.tensor([-0.6, 0.4, 0.397])
        MODULE.compute_commands(env)
        torch.testing.assert_close(env.commands, env.command_targets)

    def test_optional_direct_heading_bypass_matches_t0_manager(self):
        env = make_env(2)
        env.command_headings = torch.tensor([1.0, 1.0])
        env.commands[:, 1] = -0.2
        env.flat_direct_command = torch.tensor([True, False])
        MODULE.update_heading_command(env)
        torch.testing.assert_close(env.commands[:, 1], torch.tensor([-0.2, 0.5]))


if __name__ == "__main__":
    unittest.main()
