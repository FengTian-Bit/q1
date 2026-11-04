"""Reward-v2 semantics, parking transitions and logging reconciliation."""

import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from test_q1_tcn import SCOPE, Config, TrainConfig, tensor_env


class RewardTests(unittest.TestCase):
    def test_reward_weights_and_disabled_terms(self):
        env = tensor_env()
        expected = dict(tracking_lin_vel=3, tracking_ang_vel=.3, parking_drift=-2,
            orientation=-2, base_height=.5, lateral_velocity=-.5, lin_vel_z=-.2,
            ang_vel_xy=-.05, action_rate=-.01, torques=-.0001, collision=-1,
            dof_pos_limits=-2, termination=-200)
        self.assertEqual(set(env.reward_scales), set(expected))
        for name, weight in expected.items():
            self.assertAlmostEqual(env.reward_scales[name], weight*env.dt)
        env.commands[:, 0] = 0
        env.base_lin_vel[:, 0] = .1
        env.commands[:, 1] = 0
        env.base_ang_vel[:, 2] = .1
        torch.testing.assert_close(env._reward_tracking_lin_vel(), torch.full((4,), math.exp(-1)))
        torch.testing.assert_close(env._reward_tracking_ang_vel(), torch.full((4,), math.exp(-.25)))

    def test_termination_is_two_points_even_at_timeout_and_logging_sums(self):
        env = tensor_env()
        env.physical_failure[:] = True
        env.time_out_buf[:] = True
        env.compute_reward()
        torch.testing.assert_close(env.reward_terms['termination'], torch.full((4,), -2.))
        torch.testing.assert_close(sum(env.reward_terms.values()), env.rew_buf)
        env.physical_failure.zero_()
        env.compute_reward()
        self.assertEqual(float(env.reward_terms['termination'].abs().sum()), 0)

    def test_parking_brakes_locks_once_and_ramps(self):
        cfg = Config().rewards
        parking = SCOPE['ParkingState'](1, 'cpu')
        position = torch.zeros(1, 3)
        command = torch.tensor([[.2, 0., .397]])
        parking.reset(torch.tensor([0]), position)
        parking.update(position, command, .1, cfg)
        position[:, 0] = 1
        command[:, 0] = 0
        parking.update(position, command, .1, cfg)
        for step in range(1, 6):
            position[:, 0] = 1+step*.1
            parking.update(position, command, .1, cfg)
        torch.testing.assert_close(parking.reference[:, 0], torch.tensor([1.5]))
        self.assertTrue(parking.locked.all())
        self.assertAlmostEqual(float(parking.braking_distance[0]), .5)
        self.assertAlmostEqual(float(parking.weight[0]), 0, places=5)
        position[:, 0] = 1.6
        for _ in range(5):
            parking.update(position, command, .1, cfg)
        self.assertAlmostEqual(float(parking.weight[0]), 1, places=5)
        self.assertAlmostEqual(float(parking.penalty(cfg)[0]), .64, places=5)
        position[:, 0] = 2
        parking.update(position, command, .1, cfg)
        torch.testing.assert_close(parking.reference[:, 0], torch.tensor([1.5]))
        self.assertAlmostEqual(float(parking.penalty(cfg)[0]), 1)
        command[:, 0] = .2
        parking.update(position, command, .1, cfg)
        self.assertEqual(float(parking.penalty(cfg)[0]), 0)

    def test_initial_stand_anchors_reset_and_partial_resets_do_not_leak(self):
        cfg = Config().rewards
        state = SCOPE['ParkingState'](2, 'cpu')
        position = torch.zeros(2, 3)
        command = torch.zeros(2, 3)
        state.reset(torch.arange(2), position)
        for _ in range(12):
            state.update(position, command, .1, cfg)
        position[:, 0] = .01
        state.update(position, command, .1, cfg)
        self.assertEqual(float(state.penalty(cfg).sum()), 0)
        position[:, 0] = .12
        state.update(position, command, .1, cfg)
        torch.testing.assert_close(state.penalty(cfg), torch.ones(2), atol=1e-6, rtol=0)
        state.reset(torch.tensor([0]), position)
        self.assertEqual(float(state.weight[0]), 0)
        self.assertEqual(float(state.reference[1, 0]), 0)
        self.assertEqual(float(state.weight[1]), 1)

    def test_rollout_term_means_reconcile_environment_reward(self):
        env = tensor_env()
        cfg = TrainConfig()
        cfg.runner.num_steps_per_env = 3
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            runner = SCOPE['TCNPpoRunner'](env, cfg, folder)
            batch, _, _, _ = runner.collect()
            self.assertAlmostEqual(sum(runner.rollout_reward_metrics.values()), float(batch['environment_rewards'].mean()), places=6)
            self.assertTrue(0 <= runner.rollout_action_clipping <= 1)
        env.step(torch.full((4, 6), 2.))
        torch.testing.assert_close(env.transition_metrics['action_clipping'], torch.ones(4))

    def test_old_checkpoint_reward_defaults_are_not_silently_upgraded(self):
        saved = dict(env=SCOPE['config_dict'](Config()), train=SCOPE['config_dict'](TrainConfig()))
        saved['env']['rewards'].pop('reward_version')
        saved['env']['rewards']['scales'].pop('parking_drift')
        saved['env']['rewards']['scales']['alive'] = 1
        saved['env']['rewards']['scales']['termination'] = -2
        cfg, _ = SCOPE['scenario_config'](saved, 'stand', 101, 8, 12)
        self.assertEqual(cfg.rewards.reward_version, 1)
        self.assertEqual(cfg.rewards.scales.parking_drift, 0)
        self.assertEqual(cfg.rewards.scales.alive, 1)

    def test_parking_metrics_exclude_braking_and_keep_failure_sample(self):
        acc = SCOPE['TrialMetrics'](1, .1, np.zeros((1,3)), (False, False, False))
        sample = dict(velocity=np.zeros((1,3)), command=np.zeros(1), pitch=np.zeros(1), roll=np.zeros(1),
            support=np.ones((1,2),bool), slip=np.zeros((1,2)), saturation=np.zeros(1), position=np.zeros((1,3)),
            force=np.zeros((1,2)), failure=np.zeros(1,bool), parking_valid=np.zeros(1,bool),
            parking_error=np.array([9.]), braking_distance=np.array([.1]), action_clipping=np.array([.5]))
        acc.update(.1, sample)
        sample['parking_valid'][:] = True
        sample['parking_error'][:] = .04
        sample['failure'][:] = True
        acc.update(.2, sample)
        row = acc.rows(101,'stand')[0]
        self.assertAlmostEqual(row['locked_parking_rms_m'], .04)
        self.assertAlmostEqual(row['locked_parking_max_m'], .04)
        self.assertAlmostEqual(row['action_clipping_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
