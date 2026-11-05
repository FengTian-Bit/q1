"""Command reference preserves inputs and steady straight rolling equivalence."""
import unittest
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from test_q1_tcn import SCOPE,tensor_env
from test_q1_flat import load_code


class CommandReferenceTests(unittest.TestCase):
    def test_recovery_distribution_uses_history_gravity_and_preserves_covariance(self):
        Model=SCOPE['ActorCriticTCN']
        base=Model(30,146,6,ablation='A0',mean_limit=.95,learn_noise=False)
        with torch.no_grad():
            base.actor[-1].weight.zero_();base.actor[-1].bias.zero_()
            base.actor[-1].bias[[2,5]]=4.
        settings=dict(start=.85,slope=.3,limit=1.35,moving_limit=2.,
                      moving_gravity_start=.08,moving_gravity_full=.14)
        recovery=Model(30,146,6,ablation='A0',mean_limit=.95,learn_noise=False,
                       braking_wheel_extension=settings)
        recovery.load_state_dict(base.state_dict())
        obs=torch.zeros(3,30);obs[:,6]=torch.tensor([1.,1.,0.]);obs[:,3]=.2
        history=obs.repeat(1,16).reshape(3,16,30)
        history[0,:,3]=.03
        original,_=base.distribution(obs,history)
        actual,_=recovery.distribution(obs,history)
        torch.testing.assert_close(actual.mean[0],original.mean[0],atol=0,rtol=0)
        self.assertGreater(float(actual.mean[1,2].detach()),1.4)
        self.assertAlmostEqual(float(actual.mean[2,2].detach()),1.35,places=6)
        torch.testing.assert_close(actual.stddev,original.stddev,atol=0,rtol=0)
        self.assertEqual(set(recovery.state_dict()),set(base.state_dict()))

    def test_motion_recovery_requires_filtered_tilt_and_aligned_demand(self):
        raw=torch.full((5,6),4.,requires_grad=True);mean=.95*raw.tanh()
        commands=torch.ones(5,2);commands[0]=0
        gravity=torch.tensor([0.,.03,.11,.2,-.2])
        settings=dict(start=.85,slope=.3,limit=1.35,moving_limit=2.,
                      moving_gravity_start=.08,moving_gravity_full=.14)
        new=SCOPE['extend_braking_wheels'](mean,raw,commands,.95,settings,gravity)
        torch.testing.assert_close(new[1],mean[1],atol=0,rtol=0)
        torch.testing.assert_close(new[4],mean[4],atol=0,rtol=0)
        self.assertAlmostEqual(float(new[0,2].detach()),1.35,places=6)
        self.assertGreater(float(new[2,2].detach()),float(mean[2,2].detach()))
        self.assertLess(float(new[2,2].detach()),float(new[3,2].detach()))
        torch.testing.assert_close(new[:,[0,1,3,4]],mean[:,[0,1,3,4]],atol=0,rtol=0)
        new.sum().backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_motion_extension_keeps_zero_command_cap_and_small_outputs(self):
        raw=torch.zeros(4,6,requires_grad=True)
        with torch.no_grad():raw[:,[2,5]]=torch.tensor([4.,-4.,4.,.1])[:,None]
        mean=.95*raw.tanh();commands=torch.zeros(4,2);commands[1:,0]=1.
        old=SCOPE['extend_braking_wheels'](mean,raw,commands,.95,dict(start=.85,slope=.3,limit=1.35))
        new=SCOPE['extend_braking_wheels'](mean,raw,commands,.95,dict(start=.85,slope=.3,limit=1.35,moving_limit=2.))
        torch.testing.assert_close(new[[0,3]],old[[0,3]],atol=0,rtol=0)
        torch.testing.assert_close(new[:,[0,1,3,4]],old[:,[0,1,3,4]],atol=0,rtol=0)
        self.assertGreater(float(new[2,2].detach()),1.4)
        torch.testing.assert_close(new[1,[2,5]],-new[2,[2,5]])
        self.assertLessEqual(float(new.detach().abs().max()),2.)
        new.sum().backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_braking_extension_preserves_small_actions_and_opens_saturated_wheels(self):
        raw=torch.zeros(6,6,requires_grad=True)
        with torch.no_grad():raw[:,[2,5]]=torch.tensor([.1,1.,1.4,2.5,-3.,3.])[:,None]
        mean=.95*raw.tanh();commands=torch.zeros(6,2);commands[-1,0]=1.
        result=SCOPE['extend_braking_wheels'](mean,raw,commands,.95,dict(start=.85,slope=.3,limit=1.35))
        torch.testing.assert_close(result[:3],mean[:3],atol=0,rtol=0)
        torch.testing.assert_close(result[-1],mean[-1],atol=0,rtol=0)
        torch.testing.assert_close(result[:,[0,1,3,4]],mean[:,[0,1,3,4]],atol=0,rtol=0)
        self.assertGreater(float(result[3,2].detach()),float(mean[3,2].detach()))
        self.assertAlmostEqual(float(result[4,2].detach()),-1.35,places=6)
        result.sum().backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_braking_headroom_applies_to_both_controllers_without_standing_bias(self):
        Model=SCOPE['ActorCriticTCN']
        model=Model(30,146,6,ablation='A0',mean_limit=.95,learn_noise=False,
            parking_policy=True,parking_blend=True,
            braking_wheel_extension=dict(start=.85,slope=.3,limit=1.35))
        with torch.no_grad():
            for network,value in [(model,3.),(model.parking_policy,.1)]:
                network.actor[-1].weight.zero_();network.actor[-1].bias.zero_()
                network.actor[-1].bias[[2,5]]=value
        original=copy.deepcopy(model.parking_policy.state_dict())
        obs=torch.zeros(2,30);obs[:,5]=-1;obs[:,29]=torch.tensor([0.,1.])
        action,_=model.act_inference(obs,obs.repeat(1,16))
        torch.testing.assert_close(action[0,[2,5]],torch.full((2,),1.35))
        torch.testing.assert_close(action[1,[2,5]],torch.full((2,),.95*torch.tensor(.1).tanh()))
        for name,value in original.items():torch.testing.assert_close(value,model.parking_policy.state_dict()[name],atol=0,rtol=0)
        with torch.no_grad():model.parking_policy.actor[-1].bias[[2,5]]=3.
        action,_=model.act_inference(obs,obs.repeat(1,16))
        torch.testing.assert_close(action[1,[2,5]],torch.full((2,),1.35))

    def test_fractional_handoff_changes_only_mix_and_preserves_network_inputs(self):
        Model=SCOPE['ActorCriticTCN']
        model=Model(30,146,6,ablation='A0',mean_limit=.95,wheel_residual=True,
                    learn_noise=False,parking_policy=True,parking_blend=True,parking_fractional_flag=True)
        moving=copy.deepcopy(model);moving.parking_policy=None
        binary=copy.deepcopy(moving);binary.parking_fractional_flag=False
        obs=torch.randn(3,30);obs[:,6:8]=0;obs[:,29]=torch.tensor([1.,.5,.1])
        history=obs.repeat(1,16)
        canonical=obs.clone();canonical[:,29]=1;canonical_history=canonical.repeat(1,16)
        motion,_=moving.distribution(obs,history)
        expected_motion,_=binary.distribution(canonical,canonical_history)
        torch.testing.assert_close(motion.mean,expected_motion.mean)
        parking,_=model.parking_policy.distribution(canonical,canonical_history)
        actual,_=model.distribution(obs,history)
        weight=torch.tensor([[1.],[.5],[.1]])
        torch.testing.assert_close(actual.mean,weight*parking.mean+(1-weight)*motion.mean)
        torch.testing.assert_close(actual.stddev,weight*parking.stddev+(1-weight)*motion.stddev)
        critic=torch.randn(3,146);critic[:,32]=obs[:,29]
        canonical_critic=critic.clone();canonical_critic[:,32]=1
        torch.testing.assert_close(model.evaluate(critic),binary.evaluate(canonical_critic))

    def test_parking_sensor_filter_uses_history_and_keeps_stable_baseline(self):
        Model=SCOPE['ActorCriticTCN']
        model=Model(30,146,6,ablation='A0',mean_limit=.95,wheel_residual=True,
                    learn_noise=False,parking_policy=True,parking_blend=True,
                    parking_settle=dict(gyro_limit_scaled=.1,wheel_limit_scaled=.15,gravity_xy_limit=.15))
        moving=copy.deepcopy(model);moving.parking_policy=None
        obs=torch.zeros(3,30);obs[:,5]=-1;obs[:,29]=1
        frames=obs[:,None,:].repeat(1,16,1)
        frames[1,:,0]=.2;frames[2,:8,0]=.2
        history=frames.flatten(1)
        normal,_=moving.distribution(obs,history)
        parking,_=model.parking_policy.distribution(obs,history)
        actual,_=model.distribution(obs,history)
        weight=torch.tensor([[1.],[0.],[.5]])
        torch.testing.assert_close(actual.mean,weight*parking.mean+(1-weight)*normal.mean)
        torch.testing.assert_close(actual.stddev,weight*parking.stddev+(1-weight)*normal.stddev)

    def test_parking_blend_keeps_braking_controller_and_interpolates_locked_handoff(self):
        Model=SCOPE['ActorCriticTCN']
        model=Model(30,146,6,ablation='A0',mean_limit=.95,wheel_residual=True,
                    learn_noise=False,parking_policy=True,parking_blend=True)
        moving=copy.deepcopy(model);moving.parking_policy=None
        obs=torch.randn(3,30);obs[:,6:8]=0;obs[:,29]=torch.tensor([1.,0.,1.])
        frames=obs[:,None,:].repeat(1,16,1)
        frames[2,:8,29]=0
        history=frames.flatten(1)
        before_obs,before_history=obs.clone(),history.clone()
        normal,_=moving.distribution(obs,history)
        parking,_=model.parking_policy.distribution(obs,history)
        actual,_=model.distribution(obs,history)
        weight=torch.tensor([[1.],[0.],[.5]])
        torch.testing.assert_close(actual.mean,weight*parking.mean+(1-weight)*normal.mean)
        torch.testing.assert_close(actual.stddev,weight*parking.stddev+(1-weight)*normal.stddev)
        torch.testing.assert_close(obs,before_obs);torch.testing.assert_close(history,before_history)

    def test_zero_commands_route_to_frozen_original_policy_in_mixed_batch(self):
        Model=SCOPE['ActorCriticTCN']
        original=Model(30,146,6,ablation='A0',mean_limit=.95)
        model=Model(30,146,6,ablation='A0',mean_limit=.95,
                    wheel_residual=True,learn_noise=False,parking_policy=True,
                    freeze_motion_backbone=False)
        model.parking_policy.load_state_dict(original.state_dict())
        obs=torch.randn(4,30);obs[:,6:8]=torch.tensor([[0.,0.],[.8,0.],[0.,0.],[0.,.2]])
        history=obs.repeat(1,16)
        old,_=original.distribution(obs,history)
        actual,_=model.distribution(obs,history)
        # Selecting a sub-batch may choose a different float32 matrix kernel.
        torch.testing.assert_close(actual.mean[[0,2]],old.mean[[0,2]],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(actual.stddev[[0,2]],old.stddev[[0,2]],atol=0,rtol=0)
        self.assertFalse(any(p.requires_grad for p in model.parking_policy.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        self.assertGreater(float((actual.mean[[1,3]]-old.mean[[1,3]]).detach().abs().max()),0)
        actual.log_prob(actual.sample()).sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.wheel_residual.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))

    def test_wheel_finetune_preserves_balance_parameters_and_zero_command(self):
        Model=SCOPE['ActorCriticTCN']
        model=Model(30,146,6,ablation='A0',mean_limit=.95,
                    wheel_residual=True,learn_noise=False)
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        obs=torch.randn(4,30);obs[:,6]=torch.tensor([0.,.8,-.8,0.])
        history=obs.repeat(1,16)
        before=model.act_inference(obs,history)[0].detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        for _ in range(5):
            optimizer.zero_grad()
            mean=model.act_inference(obs,history)[0]
            loss=(mean[:,[2,5]]-obs[:,6:7]*.5).square().mean()
            loss.backward();optimizer.step()
        after=model.act_inference(obs,history)[0]
        torch.testing.assert_close(after[[0,3]],before[[0,3]],atol=0,rtol=0)
        self.assertGreater(float((after[1:3]-before[1:3]).detach().abs().max()),0)
        for name,p in model.named_parameters():
            if name in frozen:torch.testing.assert_close(p,frozen[name],atol=0,rtol=0)

    def test_new_head_warm_start_extends_optimizer_without_resetting_old_moments(self):
        scope=dict(SCOPE);load_code('tools/q1_reward_iteration.py',scope)
        Model=SCOPE['ActorCriticTCN'];PPO=SCOPE['PPO']
        source=Model(30,146,6,ablation='A0');old=PPO(source)
        sum(p.square().mean() for p in source.parameters()).backward();old.optimizer.step()
        new=Model(30,146,6,ablation='A0',wheel_residual=True,learn_noise=False,parking_policy=True)
        runner=SimpleNamespace(model=new,alg=PPO(new),cfg=SimpleNamespace(algorithm=SimpleNamespace(learning_rate=3e-5)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path=Path(folder)/'model.pt'
            torch.save(dict(model_state_dict=source.state_dict(),optimizer_state_dict=old.optimizer.state_dict(),config={'env':{}}),path)
            scope['warm_start_weights'](new,path)
            scope['inherit_optimizer'](runner,path)
        for index,state in old.optimizer.state_dict()['state'].items():
            actual=runner.alg.optimizer.state_dict()['state'][index]
            torch.testing.assert_close(actual['exp_avg'],state['exp_avg'])
            torch.testing.assert_close(actual['exp_avg_sq'],state['exp_avg_sq'])
        self.assertEqual(len(runner.alg.optimizer.state),len(old.optimizer.state))

    def test_headroom_changes_only_wheel_bound_and_keeps_leg_limit(self):
        env=tensor_env(2)
        env.cfg.normalization.clip_actions=1.4
        env.cfg.normalization.leg_clip_actions=1.
        env.step(torch.full((2,6),2.))
        torch.testing.assert_close(env.actions[:,[0,1,3,4]],torch.ones(2,4))
        torch.testing.assert_close(env.actions[:,[2,5]],torch.full((2,2),1.4))
        torch.testing.assert_close(env.raw_action_clipping,torch.ones(2))

    def test_integrated_distribution_uses_physical_actions_and_preserves_low_commands(self):
        torch.manual_seed(2)
        Model=SCOPE['ActorCriticTCN']
        cfg=dict(limit=.375,command_scale=2.,wheel_radius=.07496,
                 wheel_action_scale=10.,wheel_velocity_obs_scale=.05)
        old=Model(30,146,6,ablation='A0',mean_limit=.95)
        new=Model(30,146,6,ablation='A0',mean_limit=.95,command_reference=cfg)
        new.load_state_dict(old.state_dict())
        obs=torch.randn(3,30);obs[:,6]=0.
        history=obs.repeat(1,16);critic=torch.randn(3,146);critic[:,9]=0.
        a,_=old.distribution(obs,history);b,_=new.distribution(obs,history)
        torch.testing.assert_close(a.mean,b.mean);torch.testing.assert_close(a.stddev,b.stddev)
        torch.testing.assert_close(old.evaluate(critic),new.evaluate(critic))
        obs[:,6]=.8;history=obs.repeat(1,16)
        frame,frames,offset=SCOPE['command_reference_inputs'](obs,history,**cfg)
        expected,_=old.distribution(frame,frames)
        physical=expected.mean.clone();physical[:,[2,5]]+=offset[:,None]
        physical.clamp_(-.95,.95)
        actual,_=new.distribution(obs,history)
        torch.testing.assert_close(actual.mean,physical)
        loss=-actual.log_prob(actual.sample()).sum(-1).mean()+new.evaluate(critic).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))

    def test_zero_offset_preserves_baseline_and_does_not_mutate_inputs(self):
        scope=dict(SCOPE);load_code('wheel_legged_gym/envs/q1_tcn/command_reference.py',scope)
        obs=torch.randn(4,30);obs[:,6]=torch.tensor([0.,.5,-.5,.7])
        history=obs.repeat(1,16)
        original=obs.clone();previous=history.clone()
        a,b,offset=scope['command_reference_inputs'](obs,history,.375,2,.07496,10,.05)
        torch.testing.assert_close(a,obs);torch.testing.assert_close(b,history)
        torch.testing.assert_close(offset,torch.zeros(4))
        torch.testing.assert_close(obs,original);torch.testing.assert_close(history,previous)

    def test_virtual_steady_motion_matches_trusted_observation_and_pd_torque(self):
        scope=dict(SCOPE);load_code('wheel_legged_gym/envs/q1_tcn/command_reference.py',scope)
        radius=.07496;extra=.225;omega=extra/radius
        base=torch.randn(2,30);base[:,6]=.375*2
        high=base.clone();high[:,6]=.6*2
        high[:,[17,20]]+=omega*.05;high[:,[23,26]]+=omega/10
        transformed,history,offset=scope['command_reference_inputs'](high,high.repeat(1,16),.375,2,radius,10,.05)
        torch.testing.assert_close(transformed,base,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(history,base.repeat(1,16),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(offset,torch.full((2,),omega/10))
        old_target=torch.tensor([6.5,6.7]);old_velocity=torch.tensor([5.,5.2])
        torch.testing.assert_close(.1*((old_target+omega)-(old_velocity+omega)),.1*(old_target-old_velocity))


if __name__=='__main__':unittest.main()
