"""Reward-v2 semantics, parking transitions and logging reconciliation."""

import copy
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from test_q1_tcn import SCOPE, Config, TrainConfig, tensor_env
from test_q1_flat import make_env, load_code


class RewardTests(unittest.TestCase):
    def test_curriculum_keeps_tensor_range_and_real_command_sampler(self):
        scope=dict(SCOPE)
        scope['__name__']='reward_iteration_tests'
        load_code('tools/q1_reward_iteration.py',scope)
        env=tensor_env(32)
        env.cfg.commands.stand_probability=0
        original=env.command_ranges['lin_vel_x']
        saved=dict(env=SCOPE['config_dict'](env.cfg))
        original_rewards=copy.deepcopy(saved['env']['rewards'])
        for limit in [.35,.4,.6]:
            scope['apply_speed_limit'](env,saved,limit)
            self.assertIs(env.command_ranges['lin_vel_x'],original)
            self.assertEqual(original.shape,(32,2))
            env._resample_commands(torch.arange(32))
            self.assertLessEqual(float(env.commands[:,0].abs().max()),limit)
            self.assertEqual(saved['env']['commands']['ranges']['lin_vel_x'],[-limit,limit])
        self.assertEqual(saved['env']['rewards'],original_rewards)

    def test_speed_curriculum_requires_ordered_complete_target_and_budget(self):
        scope=dict(SCOPE)
        scope['__name__']='reward_iteration_tests'
        load_code('tools/q1_reward_iteration.py',scope)
        validate=scope['speed_curriculum']
        schedule={'speed_curriculum':[[0,.35],[200,.4],[400,.45],[600,.5],[800,.55],[1000,.6]]}
        self.assertEqual(validate(schedule,[-.6,.6],1500)[-1],(1000,.6))
        self.assertEqual(validate({},[-.6,.6],1500),[])
        for stages in [[[1,.35],[200,.6]],[[0,.35],[1000,.6]],
                       [[0,.35],[200,.7]],[[0,.35],[200,.5]],[[0,.4],[200,.35],[400,.6]]]:
            with self.assertRaises(ValueError):
                validate({'speed_curriculum':stages},[-.6,.6],1000)

    def test_speed_extension_preserves_frozen_settings_and_cli_budget(self):
        scope=dict(SCOPE)
        scope['__name__']='reward_iteration_tests'
        load_code('tools/q1_reward_iteration.py',scope)
        cfg,ppo=Config(),TrainConfig()
        ppo.ablation='A0'
        cfg.odometry.enabled=True
        cfg.rewards.scales.orientation=-20
        ppo.policy.mean_limit=.95
        SCOPE['validate_config'](cfg,ppo)
        saved=dict(env=SCOPE['config_dict'](cfg),train=SCOPE['config_dict'](ppo))
        scope['load_checkpoint']=lambda _:dict(config=copy.deepcopy(saved))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path=Path(folder)/'speed.json'
            path.write_text('{"env":{"commands":{"ranges":{"lin_vel_x":[-0.6,0.6]}}}}')
            args=SimpleNamespace(inherit_checkpoint_config=True,warm_start='frozen.pt',
                spec=path,num_envs=512,seed=7,max_iterations=1500)
            new_cfg,new_ppo,_=scope['training_config'](args)
        expected=copy.deepcopy(saved)
        expected['env']['commands']['ranges']['lin_vel_x']=[-.6,.6]
        expected['env']['env']['num_envs']=512
        expected['env']['seed']=expected['train']['seed']=7
        expected['train']['runner']['max_iterations']=1500
        expected['train']['runner']['save_interval']=100
        self.assertEqual(SCOPE['config_dict'](new_cfg),expected['env'])
        self.assertEqual(SCOPE['config_dict'](new_ppo),expected['train'])
        self.assertEqual(saved['env']['commands']['ranges']['lin_vel_x'],[-.3,.3])

    def test_a0_does_not_run_velocity_estimator(self):
        model=SCOPE['ActorCriticTCN'](27,143,6,ablation='A0')
        def forbidden(*args):
            raise AssertionError('A0 must not invoke the velocity head')
        model.velocity_head.forward=forbidden
        action, estimates=model.act_inference(torch.zeros(2,27),torch.zeros(2,432))
        self.assertEqual(action.shape,(2,6))
        self.assertEqual(float(estimates['velocity'].abs().sum()),0)

    def test_odometry_layout_and_no_true_pose_or_velocity_input(self):
        cfg=Config()
        cfg.odometry.enabled=True
        cfg.noise.add_noise=False
        cfg.init_state.reset_lin_vel=0
        cfg.init_state.reset_ang_vel=0
        SCOPE['validate_config'](cfg,TrainConfig())
        env=make_env(4,task_class=SCOPE['Q1TCN'],cfg_class=lambda:cfg)
        env.commands[:, :2]=0
        env.base_ang_vel.zero_()
        env.check_termination();env.compute_observations()
        self.assertEqual(env.obs_buf.shape,(4,30))
        self.assertEqual(env.privileged_obs_buf.shape,(4,146))
        env.dof_pos[:,env.wheel_dof_indices]+=.1
        env.check_termination();env.compute_observations()
        self.assertTrue((env.obs_buf[:,27]>.1).all())
        original=env.obs_buf.clone()
        env.root_states[:,0]+=10
        env.base_lin_vel[:,0]=123
        env.check_termination();env.compute_observations()
        torch.testing.assert_close(env.obs_buf,original,atol=1e-5,rtol=0)
        env.reset_idx(torch.tensor([0]))
        env.compute_observations()
        self.assertEqual(float(env.obs_buf[0,27:].abs().sum()),0)

    def test_odometry_warm_start_preserves_actions_and_values(self):
        scope=dict(SCOPE)
        scope['__name__']='reward_iteration_tests'
        load_code('tools/q1_reward_iteration.py',scope)
        old=SCOPE['ActorCriticTCN'](27,143,6,ablation='A0',mean_limit=.95)
        new=SCOPE['ActorCriticTCN'](30,146,6,ablation='A0',mean_limit=.95)
        scope['load_checkpoint']=lambda _:dict(model_state_dict=old.state_dict())
        scope['warm_start_weights'](new,None)
        obs=torch.randn(4,27)
        history=torch.randn(4,16,27)
        critic=torch.randn(4,143)
        extended=torch.cat((obs,torch.randn(4,3)),dim=-1)
        extended_history=torch.cat((history,torch.randn(4,16,3)),dim=-1)
        extended_critic=torch.cat((critic[:,:30],torch.randn(4,3),critic[:,30:]),dim=-1)
        torch.testing.assert_close(old.act_inference(obs,history)[0],
                                   new.act_inference(extended,extended_history)[0])
        torch.testing.assert_close(old.evaluate(critic),new.evaluate(extended_critic))

    def test_optional_policy_bounds_and_fixed_noise(self):
        model = SCOPE['ActorCriticTCN'](27,143,6,init_noise_std=.2,max_noise_std=.3,
                                      mean_limit=.95,learn_noise=False)
        model.log_std.data.fill_(100)
        model.constrain_distribution_parameters()
        distribution,_=model.distribution(torch.randn(4,27),torch.randn(4,432))
        self.assertLessEqual(float(distribution.mean.detach().abs().max()),.95)
        self.assertLessEqual(float(distribution.stddev.detach().max()),.300001)
        self.assertFalse(model.log_std.requires_grad)

    def test_optional_parking_velocity_is_gated_and_has_deadband(self):
        env=tensor_env()
        env.base_lin_vel.zero_()
        env.base_lin_vel[:,0]=.011
        env.parking.weight.zero_()
        self.assertEqual(float(env._reward_parking_velocity().sum()),0)
        env.parking.weight.fill_(1)
        torch.testing.assert_close(env._reward_parking_velocity(),torch.full((4,),.5))
        env.base_lin_vel[:,0]=.0005
        self.assertEqual(float(env._reward_parking_velocity().sum()),0)

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
