"""Wheel action unit transfer must preserve latent, physical targets and value."""
import copy
import unittest
import torch

from test_q1_tcn import SCOPE
from test_q1_flat import load_code


class WheelScaleTransferTests(unittest.TestCase):
    def test_transfer_preserves_outputs_with_changed_action_history_units(self):
        scope=dict(SCOPE)
        load_code('wheel_legged_gym/envs/q1_tcn/transfer.py',scope)
        Model=SCOPE['ActorCriticTCN']
        old=Model(30,146,6,ablation='A0',mean_limit=.95)
        new=Model(30,146,6,ablation='A0',mean_limit=.95,
                  action_output_scale=[1,1,.5,1,1,.5])
        original=copy.deepcopy(old.state_dict())
        new.load_state_dict(scope['remap_wheel_action_history'](original,30,2.))
        obs=torch.randn(4,30)
        history=torch.randn(4,16,30)
        critic=torch.randn(4,146)
        new_obs,new_history,new_critic=obs.clone(),history.clone(),critic.clone()
        new_obs[:,[23,26]]/=2
        new_history[:,:,[23,26]]/=2
        new_critic[:,[26,29,35,38,41,44]]/=2
        old_distribution,old_estimates=old.distribution(obs,history)
        new_distribution,new_estimates=new.distribution(new_obs,new_history)
        torch.testing.assert_close(new_estimates['latent'],old_estimates['latent'])
        torch.testing.assert_close(new_distribution.mean[:,[0,1,3,4]],old_distribution.mean[:,[0,1,3,4]])
        torch.testing.assert_close(new.evaluate(new_critic),old.evaluate(critic))
        # New wheel output mean/std gain .5 and actuator scale 20 reproduce
        # old physical target mean/std at actuator scale 10.
        torch.testing.assert_close(new_distribution.mean[:,[2,5]]*20,
                                   old_distribution.mean[:,[2,5]]*10)
        torch.testing.assert_close(new_distribution.stddev[:,[2,5]]*20,
                                   old_distribution.stddev[:,[2,5]]*10)
        for key,value in original.items():
            torch.testing.assert_close(value,old.state_dict()[key])
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))


if __name__=='__main__':
    unittest.main()
