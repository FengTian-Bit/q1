"""Dense 22 cm course diagnostics; tensor tests do not model contact dynamics."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_q1_ctbc import NAMESPACE
from test_q1_ctbc_high import make_high_env, high_config
from test_q1_flat import load_code


class CTBCDiagnosticTests(unittest.TestCase):
    def contact(self, env):
        env.contact_forces[:, env.feet_indices[0], 0] = 13.0
        bottom = torch.zeros(env.num_envs, 2)
        support = torch.ones(env.num_envs, 2, dtype=torch.bool)
        return bottom, support

    def tick(self, env, bottom, support, count=1):
        for _ in range(count):
            env._update_stall_diagnostics(bottom, support)

    def test_stall_requires_contact_and_three_seconds_and_never_terminates(self):
        env = make_high_env(1)
        bottom, support = torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
        self.tick(env, bottom, support, 400)
        self.assertFalse(env.ctbc_stalled.any())
        bottom, support = self.contact(env)
        self.tick(env, bottom, support)  # First contact anchors the progress clock.
        self.tick(env, bottom, support, 299)
        self.assertFalse(env.ctbc_stalled.any())
        self.tick(env, bottom, support)
        self.assertTrue(env.ctbc_stalled.all())
        env.check_termination()
        self.assertFalse(env.reset_buf.any())
        self.assertFalse(env.physical_failure.any())
        self.assertNotIn("ctbc_stall", env.reward_names)
        before = env.compute_proprioception_observations().clone()
        env.ctbc_stalled[:] = False
        env.ctbc_stall_seen[:] = False
        torch.testing.assert_close(before, env.compute_proprioception_observations())

    def test_forward_progress_or_new_supported_height_clears_stall(self):
        env = make_high_env(2)
        bottom, support = self.contact(env)
        self.tick(env, bottom, support, 301)
        self.assertTrue(env.ctbc_stalled.all())
        env.root_states[0, 0] += .04
        bottom[1, 0] = .025
        self.tick(env, bottom, support)
        self.assertFalse(env.ctbc_stalled.any())
        self.assertTrue(env.ctbc_stall_seen.all())
        # Holding a higher support is not repeated progress.
        self.tick(env, bottom, support, 300)
        self.assertTrue(env.ctbc_stalled.all())
        env.commands[0, 0] = 0
        env.ctbc_step_counts[1] = 0
        self.tick(env, bottom, support)
        self.assertFalse(env.ctbc_stalled.any())

    def test_airborne_oscillation_cannot_reset_stall_clock(self):
        env = make_high_env(1)
        bottom, support = self.contact(env)
        self.tick(env, bottom, support)
        support[:, 0] = False
        for frame in range(300):
            bottom[:, 0] = .12 if frame % 2 else .0
            self.tick(env, bottom, support)
        self.assertTrue(env.ctbc_stalled.all())

    def test_raw_and_guided_clipping_are_separate_and_do_not_modify_ppo_sample(self):
        env = make_high_env(1)
        env.metric_steps.zero_()
        env.set_guidance_mode("on")
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration/2-env.dt
        actions = torch.tensor([[1.0, -1.0, 2.0, 0.0, 0.0, 0.0]])
        saved = actions.clone()
        env.step(actions)
        torch.testing.assert_close(actions, saved)
        torch.testing.assert_close(env.ctbc_policy_clip_sum, torch.tensor([[0.,0.,1.,0.,0.,0.]]))
        torch.testing.assert_close(env.ctbc_composite_clip_sum, torch.tensor([[1.,1.,0.,0.,0.,0.]]))
        env.reset_idx(torch.tensor([0]))
        metrics = env.extras["episode"]
        self.assertEqual(metrics["ctbc_policy_clip_L_foot_joint"].item(), 1)
        self.assertEqual(metrics["ctbc_composite_clip_L_thigh_joint"].item(), 1)
        self.assertFalse(env.ctbc_policy_clip_sum.any())
        self.assertFalse(env.ctbc_composite_clip_sum.any())

    def test_per_height_success_captures_old_course_before_reset_and_keeps_denominators(self):
        env = make_high_env(3)
        env.cfg.terrain.curriculum = True
        env.terrain_levels[:] = torch.tensor([0,0,11])
        env._refresh_climb_targets()
        env.metric_steps[:] = 10
        env.ctbc_success[:] = torch.tensor([True,False,True])
        env.ctbc_stall_seen[:] = torch.tensor([False,True,False])
        env.ctbc_attempts[:] = 9
        env.ctbc_wins[:] = 9
        env.reset_idx(torch.arange(3))
        event, metrics = env.extras["ctbc_episode"], env.extras["episode"]
        self.assertEqual(event["terrain_level"].tolist(), [0,0,11])
        self.assertEqual(env.terrain_levels.tolist(), [1,1,11])
        self.assertEqual(metrics["ctbc_stair_success_h040mm"].tolist(), [1.,0.])
        self.assertEqual(metrics["ctbc_stair_success_h220mm"].tolist(), [1.])
        self.assertEqual(metrics["ctbc_stall_h040mm"].tolist(), [0.,1.])
        self.assertFalse(env.ctbc_stall_seen.any())
        self.assertFalse(env.ctbc_contact_seen.any())

    def test_ten_episode_course_requires_eight_wins_and_demotes_below_three(self):
        env = make_high_env(3)
        env.terrain_levels[:] = 1
        env._refresh_climb_targets()
        env.metric_steps[:] = 10
        for episode in range(10):
            env.ctbc_success[:] = torch.tensor([episode < 8, episode < 2, episode < 3])
            env._update_terrain_curriculum(torch.arange(3))
            if episode < 9:
                self.assertEqual(env.terrain_levels.tolist(), [1,1,1])
        self.assertEqual(env.terrain_levels.tolist(), [2,0,1])
        self.assertFalse(env.ctbc_attempts.any())

    def test_runner_logs_per_joint_noise_and_episode_weighted_height_success(self):
        namespace = dict(NAMESPACE, VecEnv=object)
        load_code("wheel_legged_gym/rsl_rl/runners/on_policy_runner.py", namespace)
        runner = object.__new__(namespace["OnPolicyRunner"])
        runner.env = SimpleNamespace(cfg=high_config(), num_envs=3)
        runner.alg = SimpleNamespace(actor_critic=SimpleNamespace(std=torch.arange(6)*.1+.2),
                                     learning_rate=.0005)
        runner.writer = Mock()
        runner.device = "cpu"
        runner.num_steps_per_env = 48
        runner.tot_timesteps = runner.tot_time = 0
        locs = dict(collection_time=1., learn_time=.1, it=0, start_iter=0, tot_iter=1,
                    mean_value_loss=0., mean_extra_loss=0., mean_surrogate_loss=0., mean_kl=0.,
                    rewbuffer=[], ep_infos=[
                        {"ctbc_stair_success_h040mm":torch.tensor([1.,0.,1.])},
                        {"ctbc_stair_success_h040mm":torch.tensor([0.])}])
        with redirect_stdout(io.StringIO()):
            runner.log(locs)
        logged = {call.args[0]:float(call.args[1]) for call in runner.writer.add_scalar.call_args_list}
        self.assertEqual(logged["Episode/ctbc_stair_success_h040mm"], .5)
        for index,name in enumerate(runner.env.cfg.asset.policy_joint_names):
            self.assertAlmostEqual(logged["Policy/noise_std_"+name], .2+index*.1, places=6)


if __name__ == "__main__":
    unittest.main()
