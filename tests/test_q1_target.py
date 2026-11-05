"""Target-mode rewards, full command coverage and physical feedforward units."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import torch
from test_q1_tcn import SCOPE,tensor_env
from test_q1_flat import load_code


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_dense_tracking_has_far_error_signal_and_keeps_zero_peak(self):
        error=torch.tensor([0.,.12,.6,1.2],requires_grad=True)
        value=SCOPE['wide_tracking_reward'](error,torch.full_like(error,.04))
        self.assertEqual(float(value[0].detach()),1.)
        self.assertTrue((value[:-1]>value[1:]).all())
        value.sum().backward()
        self.assertGreater(float(error.grad[-1].abs()),.9)
        env=tensor_env(4);env.cfg.rewards.reward_version=6
        env.cfg.rewards.tracking_sigma=.04
        env.commands[:,0]=1.2;env.base_lin_vel[:,0]=torch.tensor([1.2,1.08,.6,0.])
        actual=env._reward_tracking_lin_vel()
        torch.testing.assert_close(actual,value.detach())

    def test_full_range_and_turning_families_are_available_immediately(self):
        ids=torch.arange(400);time=torch.full((400,),3.)
        function=SCOPE['target_sequence_commands']
        vx,wz=function(time,ids,1.2,1.,.2,.25,'turning')
        self.assertTrue(((vx==0)&(wz.abs()>.1)).any())
        self.assertTrue(((wz==0)&(vx.abs()>.1)).any())
        for sv,sw in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            self.assertTrue(((vx*sv>0)&(wz*sw>0)).any())
        self.assertAlmostEqual(float(vx.abs().max()),1.2,places=5)
        self.assertAlmostEqual(float(wz.abs().max()),1.,places=5)
        a,b=function(torch.full_like(time,7.),ids,1.2,1.,.2,.25,'turning')
        torch.testing.assert_close(a,torch.zeros_like(a));torch.testing.assert_close(b,torch.zeros_like(b))
        env=tensor_env(32);env.cfg.commands.target_mode=True
        env.cfg.commands.target_stage='turning';env.cfg.commands.sequence_fraction=1.
        env.cfg.commands.ranges.lin_vel_x=[-1.2,1.2];env.cfg.commands.ranges.ang_vel_yaw=[-1.,1.]
        env.external_commands=True;env.commands[:,:2]=.123
        env._apply_sequence_commands();torch.testing.assert_close(env.commands[:,:2],torch.full((32,2),.123))

    def test_feedforward_gain_shifts_actions_but_not_physical_velocity_frame(self):
        function=SCOPE['command_reference_inputs']
        obs=torch.zeros(2,30);obs[:,6]=torch.tensor([2.4,-2.4])
        kwargs=dict(limit=.375,command_scale=2.,wheel_radius=.07496,wheel_action_scale=10.,wheel_velocity_obs_scale=.05)
        a,h,x=function(obs,obs.repeat(1,16),**kwargs)
        b,j,y=function(obs,obs.repeat(1,16),feedforward_gain=1.5,**kwargs)
        torch.testing.assert_close(a[:,[17,20]],b[:,[17,20]])
        torch.testing.assert_close(b[:,[23,26]],a[:,[23,26]]*1.5)
        torch.testing.assert_close(y,x*1.5)
        Model=SCOPE['ActorCriticTCN']
        ref=Model(30,146,6,ablation='A0',mean_limit=.95,command_reference=kwargs)
        new=Model(30,146,6,ablation='A0',mean_limit=.95,command_reference=dict(kwargs,feedforward_gain=1.5))
        new.load_state_dict(ref.state_dict());critic=torch.zeros(2,146);critic[:,9]=obs[:,6]
        # Critic real velocity translation must not be multiplied by the actuator gain.
        captured=[];hook=new.critic[0].register_forward_pre_hook(lambda m,inputs:captured.append(inputs[0].detach()))
        new.evaluate(critic);hook.remove()
        torch.testing.assert_close(captured[0][:,0],torch.tensor([-1.65,1.65]))

    def test_yaw_feedforward_is_differential_and_zero_yaw_is_exact(self):
        Model=SCOPE['ActorCriticTCN'];old=Model(30,146,6,ablation='A0',mean_limit=.95,learn_noise=False,wheel_residual=True)
        new=Model(30,146,6,ablation='A0',mean_limit=.95,learn_noise=False,wheel_residual=True,
            wheel_residual_yaw_scale=4.,steering_feedforward=dict(command_scale=.25,wheel_y=[.15,-.15],wheel_radius=.075,wheel_action_scale=10.,gain=1.5))
        new.load_state_dict(old.state_dict());obs=torch.zeros(2,30);obs[:,7]=torch.tensor([0.,.25]);history=obs.repeat(1,16)
        before,_=old.distribution(obs,history);after,_=new.distribution(obs,history)
        torch.testing.assert_close(before.mean[0],after.mean[0],atol=0,rtol=0)
        torch.testing.assert_close(after.mean[1,[2,5]]-before.mean[1,[2,5]],torch.tensor([-.3,.3]))

    def test_high_speed_retains_opposite_starting_action_and_low_speed_mapping(self):
        Model=SCOPE['ActorCriticTCN']
        ref=dict(limit=.375,command_scale=2.,wheel_radius=.07496,wheel_action_scale=10.,wheel_velocity_obs_scale=.05,feedforward_gain=1.5,action_limit=3.)
        old=Model(30,146,6,ablation='A0',mean_limit=.95,command_reference=ref)
        new=Model(30,146,6,ablation='A0',mean_limit=.95,command_reference=ref,
            motion_wheel_gain=dict(start_speed=.6,full_speed=1.2,gain=2.,command_scale=2.))
        with torch.no_grad():
            old.actor[-1].weight.zero_();old.actor[-1].bias.zero_();old.actor[-1].bias[[2,5]]=-10.
        new.load_state_dict(old.state_dict())
        obs=torch.zeros(2,30);obs[:,6]=torch.tensor([1.2,2.4]);history=obs.repeat(1,16)
        before,_=old.distribution(obs,history);after,_=new.distribution(obs,history)
        torch.testing.assert_close(before.mean[0],after.mean[0],atol=0,rtol=0)
        self.assertGreater(float(before.mean[1,2].detach()),0.)
        self.assertLess(float(after.mean[1,2].detach()),0.)

    def test_yaw_initialization_preserves_straight_policy_and_clears_adam_columns(self):
        scope=dict(SCOPE)
        load_code('wheel_legged_gym/envs/q1_tcn/target_train.py',scope)
        model=SCOPE['ActorCriticTCN'](30,146,6,ablation='A0',wheel_residual=True,learn_noise=False)
        optimizer=torch.optim.Adam(model.parameters())
        parameters=[(model.input_projection.weight,7),(model.actor[0].weight,7),
                    (model.critic[0].weight,10),(model.wheel_residual[0].weight,7)]
        for parameter,_ in parameters:
            optimizer.state[parameter]={'exp_avg':torch.ones_like(parameter),
                                        'exp_avg_sq':torch.ones_like(parameter)}
        obs=torch.randn(4,30);obs[:,7]=0
        history=obs.repeat(1,16)
        critic=torch.randn(4,146);critic[:,10]=0
        before=model.act_inference(obs,history)[0].detach()
        values=model.evaluate(critic).detach()
        scope['initialize_yaw_inputs'](SimpleNamespace(model=model,alg=SimpleNamespace(optimizer=optimizer)))
        torch.testing.assert_close(model.act_inference(obs,history)[0],before,atol=0,rtol=0)
        torch.testing.assert_close(model.evaluate(critic),values,atol=0,rtol=0)
        for parameter,column in parameters:
            self.assertEqual(int(torch.count_nonzero(parameter[:,column])),0)
            for moment in optimizer.state[parameter].values():
                self.assertEqual(int(torch.count_nonzero(moment[:,column])),0)
                self.assertTrue((moment[:,column-1]==1).all())

    def test_evaluation_covers_both_turning_quadrants_and_relative_tolerances(self):
        scope=dict(SCOPE)
        load_code('wheel_legged_gym/envs/q1_tcn/target_evaluate.py',scope)
        cases=scope['cases']('turning')
        self.assertIn(('combined120_100',1.2,1.),cases)
        self.assertIn(('combined120_m100',1.2,-1.),cases)
        self.assertEqual(scope['command_at'](3.,1.2,-1.),(1.2,-1.))
        self.assertEqual(scope['command_at'](10.,1.2,-1.),(-1.2,1.))
        self.assertEqual(scope['command_at'](7.,1.2,-1.),(0.,0.))
        self.assertEqual(scope['limits'](1.2,1.),(.12,.1))
        self.assertEqual(scope['limits'](0.,0.),(.05,.10))

    def test_training_resets_environment_before_collecting_first_rollout(self):
        scope=dict(SCOPE,hashlib=hashlib,time=SimpleNamespace(time=lambda:0.))
        load_code('wheel_legged_gym/envs/q1_tcn/target_train.py',scope)
        cfg=SCOPE['Q1TCNCfg']();ppo=SCOPE['Q1TCNCfgPPO']()
        cfg.commands.target_mode=True;cfg.commands.target_stage='linear'
        cfg.commands.ranges.lin_vel_x=[-1.2,1.2]
        ppo.runner.max_iterations=1
        parking=torch.nn.Linear(1,1).requires_grad_(False)
        env=SimpleNamespace(num_envs=2,viewer=None,sim=None,reset_called=False)
        env.reset=lambda:setattr(env,'reset_called',True)
        env.gym=SimpleNamespace(destroy_sim=lambda sim:None)

        class Runner:
            def __init__(self,env,config,output,device):
                self.env=env;self.iteration=0;self.total_steps=0;self.output=output
                self.output.mkdir()
                self.model=SimpleNamespace(parking_policy=parking)
            def save(self):
                path=self.output/('model_{}.pt'.format(self.iteration))
                path.write_bytes(b'checkpoint')
                return path
            def learn(self,iterations):
                if not self.env.reset_called:
                    raise AssertionError('First rollout saw uninitialized state and history')
                self.iteration+=iterations

        with tempfile.TemporaryDirectory(dir=os.environ.get('Q1_TEST_TMP')) as directory:
            root=Path(directory);warm=root/'warm.pt';warm.write_bytes(b'warm')
            output=root/'run'
            argv=['target_train','--stage','linear','--spec',str(root/'spec.json'),
                  '--warm_start',str(warm),'--output_dir',str(output)]
            scope.update(sys=sys,
                get_args=lambda **kw:SimpleNamespace(sim_device='cpu',rl_device='cpu'),
                training_config=lambda args:(cfg,ppo,{}),
                task_registry=SimpleNamespace(make_env=lambda *a,**kw:(env,None)),
                TCNPpoRunner=Runner,warm_start_weights=lambda *a:None,inherit_optimizer=lambda *a:None,
                evaluate=lambda *a:dict(passed=False,cases=[]))
            with patch.object(sys,'argv',argv):
                scope['main']()
            manifest=json.loads((output/'initialization.json').read_text())
            self.assertTrue(manifest['environment_reset_before_first_rollout'])

    def test_command_acceleration_limits_ramps_stops_and_direction_reversal(self):
        limit=SCOPE['rate_limit_linear_command']
        value=torch.tensor([0.])
        for _ in range(60):
            previous=value.clone();value=limit(value,torch.tensor([1.2]),.01,2.)
            self.assertLessEqual(float((value-previous).abs()),.020001)
        torch.testing.assert_close(value,torch.tensor([1.2]),atol=1e-6,rtol=0)
        for _ in range(120):
            previous=value.clone();value=limit(value,torch.tensor([-1.2]),.01,2.)
            self.assertLessEqual(float((value-previous).abs()),.020001)
        torch.testing.assert_close(value,torch.tensor([-1.2]),atol=2e-6,rtol=0)
        for _ in range(61):value=limit(value,torch.zeros(1),.01,2.)
        torch.testing.assert_close(value,torch.zeros(1),atol=0,rtol=0)
        torch.testing.assert_close(limit(value,torch.tensor([1.2]),.01,0.),torch.tensor([1.2]))

    def test_external_commands_use_same_limiter_without_double_advancement(self):
        env=tensor_env(2);env.cfg.commands.max_linear_acceleration=2.
        env.cfg.terrain.measure_heights=False;env.external_commands=True
        env.reset_idx(torch.arange(2))
        env.set_command_targets(1.2,.3)
        torch.testing.assert_close(env.commands[:,0],torch.full((2,),.02))
        torch.testing.assert_close(env.commands[:,1],torch.full((2,),.3))
        env._post_physics_step_callback()
        torch.testing.assert_close(env.commands[:,0],torch.full((2,),.02))
        env.set_command_targets(1.2,.3)
        torch.testing.assert_close(env.commands[:,0],torch.full((2,),.04))
        env.reset_idx(torch.tensor([1]))
        self.assertEqual(float(env.commands[1,0]),0.)
        self.assertAlmostEqual(float(env.commands[0,0]),.04,places=6)

    def test_random_training_target_persists_until_resampling(self):
        env=tensor_env(2);env.cfg.commands.max_linear_acceleration=2.
        env.cfg.terrain.measure_heights=False;env.cfg.commands.sequence_fraction=0.
        env.reset_idx(torch.arange(2))
        env.episode_length_buf[:]=1
        env.command_targets[:]=env.commands
        env.command_targets[:,0]=1.2
        for i in range(1,11):
            env._post_physics_step_callback()
            env.episode_length_buf+=1
            torch.testing.assert_close(env.commands[:,0],torch.full((2,),i*.02))
            torch.testing.assert_close(env.command_targets[:,0],torch.full((2,),1.2))

    def test_relative_tracking_treats_ten_percent_errors_equally(self):
        env=tensor_env(3);cfg=env.cfg.rewards;cfg.reward_version=7
        cfg.relative_tracking=dict(fraction=.1,linear_floor=.03,yaw_floor=.05,zero_yaw_width=.1)
        env.commands[:,:2]=0
        env.commands[:,0]=torch.tensor([.3,.6,1.2])
        env.base_lin_vel[:,0]=env.commands[:,0]*.9
        expected=torch.full((3,),torch.exp(torch.tensor(-1.)).item()-.1)
        torch.testing.assert_close(env._reward_tracking_lin_vel(),expected,atol=2e-6,rtol=0)
        env.commands[:,0]=.6;env.commands[:,1]=torch.tensor([0.,.5,1.])
        env.base_ang_vel[:,2]=torch.tensor([.1,.45,.9])
        torch.testing.assert_close(env._reward_tracking_ang_vel(),expected,atol=2e-6,rtol=0)

    def test_wheel_differential_guides_correct_turns_only_with_ground_support(self):
        env=tensor_env(7)
        env.cfg.rewards.wheel_diff_tracking=dict(track_width=.299,sigma=1.)
        env.commands[:,1]=torch.tensor([1.,-1.,1.,0.,1.,1.,1.])
        env.contact_forces[:,env.feet_indices,2]=10.
        target=.299/env.cfg.asset.wheel_radius
        wheel=torch.tensor([[-target/2,target/2],[target/2,-target/2],
                            [target/2,-target/2],[-target/2,target/2],
                            [-target/2,target/2],[16-target/2,16+target/2],[0.,0.]])
        for index,column in enumerate([2,5]):
            env.dof_vel[:,env.policy_dof_indices[column]]=wheel[:,index]
        env.contact_forces[4,env.feet_indices[0],2]=0.
        actual=env._reward_tracking_wheel_diff()
        expected=torch.tensor([1.,1.,torch.exp(torch.tensor(-4.)).item(),0.,0.,1.,
                               torch.exp(torch.tensor(-1.)).item()])
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        self.assertGreater(float(actual[6]),float(actual[2]))
        env.commands[:,1]=0.
        torch.testing.assert_close(env._reward_tracking_wheel_diff(),torch.zeros(7),atol=0,rtol=0)

    def test_wheel_guidance_requires_explicit_valid_geometry(self):
        env=tensor_env(2);config=SCOPE['Q1TCNCfgPPO']()
        env.cfg.rewards.scales.tracking_wheel_diff=.5
        with self.assertRaises(ValueError):SCOPE['validate_config'](env.cfg,config)
        for settings in [dict(track_width=0.,sigma=1.),dict(track_width=.299,sigma=float('nan'))]:
            env.cfg.rewards.wheel_diff_tracking=settings
            with self.assertRaises(ValueError):SCOPE['validate_config'](env.cfg,config)
        env.cfg.rewards.wheel_diff_tracking=dict(track_width=.299,sigma=1.)
        SCOPE['validate_config'](env.cfg,config)

    def test_relative_tracking_preserves_idle_rewards_and_rejects_bad_config(self):
        env=tensor_env(2);cfg=env.cfg.rewards
        cfg.reward_version=7
        cfg.relative_tracking=dict(fraction=.1,linear_floor=.03,yaw_floor=.05,zero_yaw_width=.1)
        env.commands[:,:2]=0;env.base_lin_vel[:,0]=.01;env.base_ang_vel[:,2]=.1
        linear=env._reward_tracking_lin_vel();yaw=env._reward_tracking_ang_vel()
        cfg.reward_version=6
        torch.testing.assert_close(env._reward_tracking_lin_vel(),linear,atol=0,rtol=0)
        torch.testing.assert_close(env._reward_tracking_ang_vel(),yaw,atol=0,rtol=0)
        with self.assertRaises(ValueError):SCOPE['validate_config'](env.cfg,SCOPE['Q1TCNCfgPPO']())
        cfg.reward_version=7;cfg.relative_tracking['fraction']=0
        with self.assertRaises(ValueError):SCOPE['validate_config'](env.cfg,SCOPE['Q1TCNCfgPPO']())


if __name__=='__main__':unittest.main()
