"""Check the ported T0 curriculum without importing Isaac Gym."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q1_t0_curriculum", ROOT / "wheel_legged_gym/envs/q1/t0_curriculum.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
terrain_levels = MODULE.terrain_levels


def make_env(levels, types=None, rows=4):
    count = len(levels)
    types = [0] * count if types is None else types
    origins = torch.zeros(rows, 2, 3)
    origins[:, :, 0] = torch.arange(rows)[:, None] * 8.0 + 4.0
    origins[:, :, 1] = torch.tensor([4.0, 12.0])
    origins[:, :, 2] = torch.arange(rows)[:, None] * 0.1
    levels = torch.tensor(levels, dtype=torch.long)
    types = torch.tensor(types, dtype=torch.long)
    selected = origins[levels, types].clone()
    env = SimpleNamespace(
        init_done=True,
        terrain=SimpleNamespace(env_length=8.0),
        max_episode_length_s=20.0,
        max_terrain_level=rows,
        terrain_levels=levels,
        terrain_types=types,
        terrain_origins=origins,
        env_origins=selected,
        root_states=torch.zeros(count, 13),
        commands=torch.zeros(count, 3),
        refreshes=[],
    )
    env.root_states[:, :3] = selected
    env.commands[:, 0] = 0.2
    env.commands[:, 2] = 0.397
    env._refresh_terrain_classes = lambda: env.refreshes.append("classes")
    env._refresh_climb_targets = lambda: env.refreshes.append("targets")
    return env


class T0CurriculumTests(unittest.TestCase):
    def test_no_curriculum_during_initial_reset_or_empty_reset(self):
        terrain_levels(SimpleNamespace(init_done=False), torch.tensor([0]))
        terrain_levels(SimpleNamespace(init_done=True), torch.tensor([], dtype=torch.long))

    def test_strict_t0_distance_thresholds_and_metadata_refresh(self):
        env = make_env([1, 1, 1, 1], types=[0, 1, 0, 1])
        env.root_states[:, 0] += torch.tensor([4.01, 4.0, 1.99, 2.0])
        terrain_levels(env, torch.arange(4))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([2, 1, 0, 1]))
        torch.testing.assert_close(env.success_ids, torch.tensor([0]))
        torch.testing.assert_close(env.fail_ids, torch.tensor([2]))
        torch.testing.assert_close(
            env.env_origins, env.terrain_origins[env.terrain_levels, env.terrain_types]
        )
        self.assertEqual(env.refreshes, ["classes", "targets"])

    def test_planar_net_displacement_includes_lateral_direction(self):
        env = make_env([1, 1])
        env.root_states[0, 1] += 4.1
        env.root_states[1, :2] += torch.tensor([3.0, 3.0])
        terrain_levels(env, torch.arange(2))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([2, 2]))

    def test_q1_yaw_and_height_commands_do_not_count_as_translation(self):
        env = make_env([1, 1])
        env.commands[:, 0] = 0.0
        env.commands[:, 1] = torch.tensor([100.0, -100.0])
        env.commands[:, 2] = 100.0
        terrain_levels(env, torch.arange(2))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([1, 1]))
        self.assertEqual(len(env.fail_ids), 0)

    def test_reverse_command_uses_absolute_translational_speed(self):
        env = make_env([1])
        env.commands[0, 0] = -0.2
        env.root_states[0, 0] -= 1.0
        terrain_levels(env, torch.tensor([0]))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([0]))

    def test_top_row_randomizes_and_bottom_row_clamps(self):
        env = make_env([3, 0, 2])
        env.root_states[0, 0] += 5.0
        env.root_states[2, 0] += 4.0
        with patch.object(MODULE.torch, "randint_like", return_value=torch.tensor([1, 3, 0])) as draw:
            terrain_levels(env, torch.arange(3))
        draw.assert_called_once()
        torch.testing.assert_close(env.terrain_levels, torch.tensor([1, 0, 2]))

    def test_update_only_finished_environments_without_success_window(self):
        env = make_env([1, 1, 1])
        before = env.env_origins.clone()
        env.root_states[:, 0] += 5.0
        # No ctbc_attempts, ctbc_wins, physical_failure or metric counters are
        # supplied: those must not gate this T0 distance curriculum.
        terrain_levels(env, torch.tensor([2]))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([1, 1, 2]))
        torch.testing.assert_close(env.env_origins[:2], before[:2])
        torch.testing.assert_close(env.success_ids, torch.tensor([2]))

    def test_flat_tile_obeys_the_same_t0_distance_curriculum(self):
        env = make_env([1], types=[0])
        env.terrain_classes = torch.tensor([-1])
        env.root_states[0, 0] += 5.0
        terrain_levels(env, torch.tensor([0]))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([2]))

    def test_expected_distance_uses_full_episode_limit(self):
        env = make_env([1])
        env.max_episode_length_s = 10.0
        env.root_states[0, 0] += 1.0
        terrain_levels(env, torch.tensor([0]))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([1]))
        env.max_episode_length_s = 20.0
        terrain_levels(env, torch.tensor([0]))
        torch.testing.assert_close(env.terrain_levels, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
