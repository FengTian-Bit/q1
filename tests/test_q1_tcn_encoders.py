"""Raw angular increments and zero-impact activation of previously empty slots."""
import copy
import unittest
import torch

from test_q1_tcn import SCOPE,Config,TrainConfig
from test_q1_flat import make_env,load_code


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fractional_ramp_preserves_initial_stand_and_full_position_feedback(self):
        cfg=Config();cfg.odometry.enabled=True;cfg.odometry.parking_blend_ramp_s=.5
        train=TrainConfig();train.policy.parking_policy=True;train.policy.parking_blend=True
        train.policy.parking_fractional_flag=True
        SCOPE['validate_config'](cfg,train)
        env=make_env(4,task_class=SCOPE['Q1TCN'],cfg_class=lambda:cfg)
        env.parking.stationary[:]=True;env.parking.locked[:]=True
        env.parking.initial_stand[:]=torch.tensor([True,False,False,False])
        env.parking.elapsed[:]=torch.tensor([0.,.5,.75,1.])
        env.odometry.xy[:]=.01
        env._parking_observation()
        expected=torch.tensor([1.,torch.finfo(torch.float32).eps,.5,1.])
        torch.testing.assert_close(env.odometry_features[:,2],expected)
        torch.testing.assert_close(env.odometry_features[:,:2],torch.full((4,2),.2))

    def test_parking_observation_delay_keeps_initial_stand_and_reward_lock(self):
        cfg=Config();cfg.odometry.enabled=True;cfg.odometry.parking_observation_delay_s=1.
        SCOPE['validate_config'](cfg,TrainConfig())
        env=make_env(4,task_class=SCOPE['Q1TCN'],cfg_class=lambda:cfg)
        env.parking.stationary[:]=torch.tensor([True,True,True,False])
        env.parking.locked[:]=True
        env.parking.initial_stand[:]=torch.tensor([True,False,False,False])
        env.parking.elapsed[:]=torch.tensor([0.,.75,1.,2.])
        locked=env.parking.locked.clone();reference=env.parking.reference.clone()
        env._parking_observation()
        torch.testing.assert_close(env.odometry_features[:,2],torch.tensor([1.,0.,1.,0.]))
        torch.testing.assert_close(env.parking.locked,locked)
        torch.testing.assert_close(env.parking.reference,reference)
        cfg.odometry.parking_observation_delay_s=0.
        env._parking_observation()
        torch.testing.assert_close(env.odometry_features[:,2],torch.tensor([1.,1.,1.,0.]))

    def test_sensor_increment_wrap_reset_and_no_body_velocity_input(self):
        cfg=Config();cfg.odometry.enabled=True;cfg.odometry.observe_wheel_increments=True
        cfg.noise.add_noise=False
        SCOPE['validate_config'](cfg,TrainConfig())
        env=make_env(4,task_class=SCOPE['Q1TCN'],cfg_class=lambda:cfg)
        env.commands[:,:2]=0;env.base_ang_vel.zero_()
        env.check_termination();env.compute_observations()
        env.dof_pos[:,env.wheel_dof_indices]+=.1
        env.check_termination();env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:,[11,14]],torch.ones(4,2),atol=1e-5,rtol=0)
        env.dof_pos[:,env.wheel_dof_indices]+=.1-2*torch.pi
        env.check_termination();env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:,[11,14]],torch.ones(4,2),atol=1e-4,rtol=0)
        env.check_termination();env.compute_observations()
        before=env.obs_buf.clone()
        env.root_states[:,0]+=100;env.base_lin_vel[:,0]=123
        env.check_termination();env.compute_observations()
        torch.testing.assert_close(env.obs_buf,before,atol=1e-5,rtol=0)
        env.reset_idx(torch.tensor([0]));env.compute_observations()
        self.assertEqual(float(env.obs_buf[0,[11,14]].abs().sum()),0)

    def test_new_channels_preserve_latent_actions_and_values_at_transfer(self):
        scope=dict(SCOPE);scope['__name__']='encoder_transfer_test'
        load_code('tools/q1_reward_iteration.py',scope)
        Model=SCOPE['ActorCriticTCN']
        old=Model(30,146,6,ablation='A0',mean_limit=.95)
        new=Model(30,146,6,ablation='A0',mean_limit=.95)
        original=copy.deepcopy(old.state_dict())
        saved=dict(model_state_dict=original,config=dict(env=dict(odometry=dict(observe_wheel_increments=False))))
        scope['load_checkpoint']=lambda _:saved
        scope['warm_start_weights'](new,None,True)
        obs=torch.randn(4,30);history=torch.randn(4,16,30);critic=torch.randn(4,146)
        obs[:,[11,14]]=0;history[:,:,[11,14]]=0;critic[:,[14,17]]=0
        new_obs,new_history,new_critic=obs.clone(),history.clone(),critic.clone()
        new_obs[:,[11,14]]=torch.randn(4,2)
        new_history[:,:,[11,14]]=torch.randn(4,16,2)
        new_critic[:,[14,17]]=torch.randn(4,2)
        old_action,old_state=old.act_inference(obs,history)
        new_action,new_state=new.act_inference(new_obs,new_history)
        torch.testing.assert_close(new_action,old_action)
        torch.testing.assert_close(new_state['latent'],old_state['latent'])
        torch.testing.assert_close(new.evaluate(new_critic),old.evaluate(critic))
        for key,value in original.items():torch.testing.assert_close(value,old.state_dict()[key])


if __name__=='__main__':unittest.main()
