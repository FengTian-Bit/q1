"""Regression cases spanning transitions and physically relevant height frames."""
import unittest
from types import SimpleNamespace

import torch

from test_q1 import improved
from test_q1_flat import NS, make_env, load_code
from test_q1_terrain import rough_env


class Q1TrainingLogicTests(unittest.TestCase):
    def test_timeout_and_episode_info_do_not_leak_to_next_transition(self):
        for create in (improved, make_env):
            with self.subTest(task=create.__name__):
                env = create(2)
                env.episode_length_buf[0] = env.max_episode_length - 1
                _, _, _, done, first, _ = env.step(torch.zeros(2, 6))
                self.assertEqual(done.tolist(), [True, False])
                self.assertEqual(first["time_outs"].tolist(), [True, False])
                self.assertIn("episode", first)
                _, _, _, done, following, _ = env.step(torch.zeros(2, 6))
                self.assertEqual(done.tolist(), [False, False])
                self.assertEqual(following["time_outs"].tolist(), [False, False])
                self.assertNotIn("episode", following)
                self.assertEqual(first["time_outs"].tolist(), [True, False])

    def test_real_ppo_bootstraps_only_the_timeout_transition(self):
        namespace = dict(NS)
        load_code("wheel_legged_gym/rsl_rl/modules/actor_critic.py", namespace)
        load_code("wheel_legged_gym/rsl_rl/storage/rollout_storage.py", namespace)
        load_code("wheel_legged_gym/rsl_rl/algorithms/ppo.py", namespace)
        ppo = namespace["PPO"].__new__(namespace["PPO"])
        ppo.gamma, ppo.device = .99, "cpu"
        captured = []
        ppo.storage = SimpleNamespace(add_transitions=lambda transition: captured.append(transition.rewards.clone()))
        ppo.actor_critic = SimpleNamespace(reset=lambda done: None)
        env = improved(2)
        env.episode_length_buf[0] = env.max_episode_length - 1
        for _ in range(2):
            ppo.transition = SimpleNamespace(values=torch.full((2, 1), 10.0), clear=lambda: None)
            obs, _, reward, done, info, _ = env.step(torch.zeros(2, 6))
            before = reward.clone()
            ppo.process_env_step(reward, done, info, obs)
            correction = captured[-1] - before
            expected = torch.tensor([9.9, 0.0]) if len(captured) == 1 else torch.zeros(2)
            torch.testing.assert_close(correction, expected)

    def test_root_crossing_riser_does_not_invent_a_fall_or_height_penalty(self):
        env = rough_env(2)
        env.external_commands = True
        env.height_samples[41:, :] = 40
        env.root_states[:, 2] = .397
        env.root_states[:, 0] = torch.tensor([4.09, 4.11])
        env.episode_length_buf[:] = 1
        env._post_physics_step_callback()
        env.check_termination()
        self.assertFalse(env.physical_failure.any())
        torch.testing.assert_close(env.base_height[0], env.base_height[1])
        torch.testing.assert_close(env._reward_base_height(), torch.ones(2), atol=1e-5, rtol=0)
        # Height still rejects an actual collapsed trunk above the same wheels.
        env.root_states[0, 2] = .15
        env._post_physics_step_callback()
        env.check_termination()
        self.assertTrue(env.physical_failure[0])
        # Actual body collision still fails independently of the height proxy.
        env.contact_forces[1, 0, 2] = 10.0
        env.check_termination()
        self.assertTrue(env.physical_failure[1])

    def test_support_height_handles_one_wheel_two_heights_and_flight(self):
        env = rough_env(2)
        env.root_states[:, 2] = .497
        wheel = env.rigid_body_states[:, env.feet_indices].clone()
        wheel[:, 0, 2] = env.cfg.asset.wheel_radius
        wheel[:, 1, 2] = env.cfg.asset.wheel_radius + .2
        env.rigid_body_states[:, env.feet_indices] = wheel
        torch.testing.assert_close(env._height_above_support(), torch.full((2,), .397), atol=1e-6, rtol=0)
        env.contact_forces[:, env.feet_indices[1]] = 0
        torch.testing.assert_close(env._height_above_support(), torch.full((2,), .497), atol=1e-6, rtol=0)
        env.contact_forces[:, env.feet_indices] = 0
        env.height_samples[41:, :] = 40
        env.root_states[:, 0] = 4.11
        # Both wheels still project to lower ground at x=4.0.
        torch.testing.assert_close(env._height_above_support(), torch.full((2,), .497), atol=1e-6, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
