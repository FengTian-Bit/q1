"""Reward-v5 migration and estimator-free PPO/export on the real T0 task."""
from copy import deepcopy
from collections import deque
import math
import statistics
import unittest

import torch

from test_q1_flat import load_code
from test_q1_t0 import NAMES, t0_env


class T0RewardV5Tests(unittest.TestCase):
    def test_v5_weights_replace_locomotion_and_preserve_ctbc(self):
        env = t0_env()
        weights = env.cfg.rewards.scales
        expected = dict(tracking_lin_vel=3., tracking_ang_vel=1.5,
                        parking_drift=-3., parking_velocity=-.5, orientation=-20.,
                        base_height=.5, lateral_velocity=-.5, lin_vel_z=-.2,
                        ang_vel_xy=-.1, action_rate=-.02, torques=-.0001,
                        collision=-1., dof_pos_limits=-2., termination=-200.)
        for name, value in expected.items():
            self.assertEqual(getattr(weights, name), value, name)
        for name in ('alive', 'nominal_state', 'dof_vel', 'dof_acc', 'action_smooth',
                     'turning_yaw_tracking', 'actual_posture', 'leg_symmetry',
                     'rolling_contact', 'rolling_leg_motion', 'rolling_slip',
                     'speed_error', 'parking_wheel_speed', 'straight_path'):
            self.assertNotIn(name, env.reward_names)
        for name in ('ctbc_target', 'ctbc_clearance', 'ctbc_contact'):
            self.assertIn(name, env.reward_names)
        self.assertFalse(NAMES['Q1T0CfgPPO']().policy.velocity_estimation_enabled)
        self.assertEqual(env.cfg.commands.ranges.lin_vel_x, [-1.2, 1.2])

    def test_yaw_and_linear_share_exact_moving_and_standing_kernel(self):
        env = t0_env(3)
        env.commands[:, :2] = torch.tensor([[.3, .4], [0., .4], [0., 0.]])
        env.base_lin_vel[:, 0] = env.commands[:, 0] + .02
        env.base_ang_vel[:, 2] = env.commands[:, 1] + .02
        expected = torch.exp(-torch.tensor([.02**2/.0025, .02**2/.0004, .02**2/.0004]))
        torch.testing.assert_close(env._reward_tracking_lin_vel(), expected)
        torch.testing.assert_close(env._reward_tracking_ang_vel(), expected)
        env.base_ang_vel[:, 2] = env.commands[:, 1]
        torch.testing.assert_close(env._reward_tracking_ang_vel(), torch.ones(3))

    def test_parking_brakes_locks_ramps_and_resets_without_penalizing_turn(self):
        env = t0_env(2)
        env.commands[:, :2] = torch.tensor([[0., 0.], [0., .4]])
        env.v5_parking.reset(torch.arange(2), env.root_states)
        for _ in range(110):
            env.check_termination()
        env.root_states[:, 0] += .035
        env.check_termination()
        torch.testing.assert_close(env._reward_parking_drift(), torch.tensor([1., 0.]))
        env.base_lin_vel.zero_()
        env.base_lin_vel[:, 0] = .021
        torch.testing.assert_close(env._reward_parking_velocity(), torch.tensor([1., 0.]))
        env.reset_idx(torch.tensor([0]))
        self.assertEqual(env.v5_parking.weight[0], 0.)
        self.assertEqual(env.v5_parking.error[0], 0.)

    def test_height_and_failure_match_v5(self):
        env = t0_env(2)
        env.base_height[:] = env.commands[:, 2] + .05
        torch.testing.assert_close(env._reward_base_height(), torch.full((2,), math.exp(-1)))
        env.physical_failure = torch.tensor([True, False])
        env.time_out_buf[:] = True
        torch.testing.assert_close(env._reward_termination(), torch.tensor([1., 0.]))

    def test_no_estimator_forward_loss_or_export_leak_during_real_ppo(self):
        scope = dict(NAMES, deepcopy=deepcopy, optim=torch.optim,
                     statistics=statistics, deque=deque)
        for path in ('modules/actor_critic.py', 'modules/actor_critic_sequence.py',
                     'storage/rollout_storage.py', 'algorithms/ppo.py', 'modules/policy_exporter.py'):
            load_code('wheel_legged_gym/rsl_rl/' + path, scope)
        env = t0_env(4)
        model = scope['ActorCriticSequence'](31, 154, 6, 155, 3,
            encoder_hidden_dims=[16, 8], actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8],
            velocity_estimation_enabled=False)
        ppo = scope['PPO'](model, num_learning_epochs=1, num_mini_batches=1)
        self.assertIsNone(ppo.extra_optimizer)
        exporter = scope['SequenceInferencePolicy'](model)
        def forbidden(*args):
            raise AssertionError('Disabled velocity estimator was executed')
        model.encoder.forward = forbidden
        exporter.encoder.forward = forbidden
        obs, history = env.obs_buf, env.obs_history
        expected, latent = model.act_inference(obs, history)
        self.assertFalse(latent.any())
        torch.testing.assert_close(exporter(obs, history), expected)
        torch.testing.assert_close(model.act_inference(obs, history+1)[0], expected)
        ppo.init_storage(4, 3, [31], [154], [155], [6])
        before = [p.clone() for p in model.encoder.parameters()]
        with torch.inference_mode():
            for _ in range(3):
                action = ppo.act(env.obs_buf, env.obs_history, env.privileged_obs_buf)
                obs, critic, reward, done, info, history = env.step(action)
                ppo.process_env_step(reward, done, info, obs)
            ppo.compute_returns(torch.cat((critic, model.encode(history)), dim=-1))
        losses = ppo.update()
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertEqual(losses[-1], 0.)
        for original, current in zip(before, model.encoder.parameters()):
            torch.testing.assert_close(current, original)


if __name__ == '__main__':
    unittest.main()
