"""Physical left/right mapping, temporal alignment and PPO symmetry gradients."""

import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch.distributions import Normal

from test_q1_tcn import SCOPE, tensor_env


class SymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physical_parities_joint_order_and_involution(self):
        mirror = SCOPE['mirror_observations']
        for width in (27, 30):
            obs = torch.arange(1, width+1, dtype=torch.float32).reshape(1, -1)
            result = mirror(obs)
            torch.testing.assert_close(result[:, :9], obs[:, :9]*torch.tensor([-1, 1, -1, 1, -1, 1, 1, -1, 1]))
            for start in (9, 15, 21):
                torch.testing.assert_close(result[:, start:start+6], obs[:, [start+3, start+4, start+5, start, start+1, start+2]])
            if width == 30:
                torch.testing.assert_close(result[:, 27:], obs[:, 27:]*torch.tensor([1, -1, 1]))
            torch.testing.assert_close(mirror(result), obs, atol=0, rtol=0)
            frames = obs.repeat(2, 16)+torch.arange(16).repeat_interleave(width)
            actual = SCOPE['mirror_history'](frames, width).reshape(2, 16, width)
            for frame in range(16):
                torch.testing.assert_close(actual[:, frame], mirror(frames.reshape(2, 16, width)[:, frame]))
        action = torch.arange(6).reshape(1, 6)
        torch.testing.assert_close(SCOPE['mirror_actions'](action), torch.tensor([[3, 4, 5, 0, 1, 2]]))

    def test_both_branches_have_gradients_and_optimization_reduces_error(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.arange(6, dtype=torch.float32)/10)
                self.means = []
            def distribution(self, obs, history):
                mean = obs[:, 9:15]+self.bias
                mean.retain_grad()
                self.means.append(mean)
                return Normal(mean, torch.ones_like(mean)), {}

        model = Policy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        obs = torch.randn(8, 30); history = obs.repeat(1, 16)
        mean = model.distribution(obs, history)[0].mean
        loss = SCOPE['policy_symmetry_loss'](model, obs, history, mean)
        before = float(loss.detach())
        self.assertGreater(before, 0)
        loss.backward()
        self.assertTrue(all(mean.grad is not None and mean.grad.abs().sum()>0 for mean in model.means))
        optimizer.step()
        mean = model.distribution(obs, history)[0].mean
        after = SCOPE['policy_symmetry_loss'](model, obs, history, mean)
        self.assertLess(float(after.detach()), before)
        with torch.no_grad():model.bias.zero_()
        mean = model.distribution(obs, history)[0].mean
        self.assertEqual(float(SCOPE['policy_symmetry_loss'](model, obs, history, mean).detach()), 0.)
        with torch.no_grad():model.bias[0]=1.
        mean = model.distribution(obs, history)[0].mean
        masked = SCOPE['policy_symmetry_loss'](model, obs, history, mean, torch.zeros(8))
        self.assertEqual(float(masked.detach()), 0.)

    def test_ppo_updates_symmetry_and_preserves_frozen_parking(self):
        env = tensor_env(4)
        env.external_commands=True;env.commands[:,0]=.3;env.compute_observations()
        config = SCOPE['Q1TCNCfgPPO']();config.ablation='A0'
        config.runner.num_steps_per_env=2
        config.algorithm.num_learning_epochs=1;config.algorithm.num_mini_batches=1
        config.algorithm.symmetry_loss_coef=.1;config.algorithm.symmetry_batch_size=3
        config.policy.parking_policy=True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            runner=SCOPE['TCNPpoRunner'](env,config,directory)
            before={key:value.clone() for key,value in runner.model.parking_policy.state_dict().items()}
            batch,last_values,_,_=runner.collect()
            result=runner.alg.update(batch,last_values)
            self.assertGreater(result['symmetry_loss'],0.)
            self.assertAlmostEqual(result['symmetry_loss_weighted'],.1*result['symmetry_loss'],places=6)
            for key,value in before.items():
                torch.testing.assert_close(value,runner.model.parking_policy.state_dict()[key],atol=0,rtol=0)
            self.assertTrue(all(p.grad is None for p in runner.model.parking_policy.parameters()))

    def test_resume_allows_only_explicit_symmetry_coefficient_update(self):
        env=tensor_env(2);config=SCOPE['Q1TCNCfgPPO']()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory)
            source=SCOPE['TCNPpoRunner'](env,config,root/'source')
            source.model.actor[0].weight.square().sum().backward()
            source.alg.optimizer.step()
            source.iteration=750;source.total_steps=147456000
            checkpoint=source.save()
            updated=copy.deepcopy(config);updated.algorithm.symmetry_loss_coef=.1
            resumed=SCOPE['TCNPpoRunner'](env,updated,root/'updated')
            with self.assertRaises(ValueError):resumed.load(checkpoint)
            resumed.load(checkpoint,allow_symmetry_update=True)
            self.assertEqual(resumed.iteration,750)
            self.assertEqual(resumed.total_steps,147456000)
            for key,value in source.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)
            original_state=source.alg.optimizer.state[source.model.actor[0].weight]
            loaded_state=resumed.alg.optimizer.state[resumed.model.actor[0].weight]
            for key in ['step','exp_avg','exp_avg_sq']:
                torch.testing.assert_close(original_state[key],loaded_state[key],atol=0,rtol=0)
            changed=copy.deepcopy(updated);changed.algorithm.learning_rate*=2
            invalid=SCOPE['TCNPpoRunner'](env,changed,root/'invalid')
            with self.assertRaises(ValueError):invalid.load(checkpoint,allow_symmetry_update=True)

    def test_tracking_update_is_explicit_and_other_rewards_stay_strict(self):
        env=tensor_env(2);env.cfg.rewards.reward_version=6
        config=SCOPE['Q1TCNCfgPPO']()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory);source=SCOPE['TCNPpoRunner'](env,config,root/'source')
            checkpoint=source.save()
            updated_env=tensor_env(2);updated_env.cfg=copy.deepcopy(env.cfg)
            updated_env.cfg.rewards.reward_version=7
            updated_env.cfg.rewards.relative_tracking=dict(fraction=.1,linear_floor=.03,yaw_floor=.05,zero_yaw_width=.1)
            resumed=SCOPE['TCNPpoRunner'](updated_env,config,root/'updated')
            with self.assertRaises(ValueError):resumed.load(checkpoint,allow_symmetry_update=True)
            resumed.load(checkpoint,allow_tracking_update=True)
            self.assertEqual(resumed.snapshot['env']['rewards']['reward_version'],7)
            updated_env.cfg.rewards.scales.orientation=-999.
            invalid=SCOPE['TCNPpoRunner'](updated_env,config,root/'invalid')
            with self.assertRaises(ValueError):invalid.load(checkpoint,allow_tracking_update=True)

    def test_yaw_weight_update_preserves_training_state_and_rejects_other_changes(self):
        env=tensor_env(2)
        env.cfg.rewards.scales.tracking_lin_vel=4.
        env.cfg.rewards.scales.tracking_ang_vel=3.
        config=SCOPE['Q1TCNCfgPPO']()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory);source=SCOPE['TCNPpoRunner'](env,config,root/'source')
            source.model.actor[0].weight.square().sum().backward()
            source.alg.optimizer.step()
            source.iteration=500;source.total_steps=98304000
            checkpoint=source.save()
            updated=tensor_env(2);updated.cfg=copy.deepcopy(env.cfg)
            updated.cfg.rewards.scales.tracking_ang_vel=2.
            resumed=SCOPE['TCNPpoRunner'](updated,config,root/'updated')
            with self.assertRaises(ValueError):resumed.load(checkpoint)
            resumed.load(checkpoint,allow_yaw_weight_update=True)
            self.assertEqual((resumed.iteration,resumed.total_steps),(500,98304000))
            self.assertEqual(resumed.snapshot['env']['rewards']['scales']['tracking_ang_vel'],2.)
            for key,value in source.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)
            original=source.alg.optimizer.state[source.model.actor[0].weight]
            loaded=resumed.alg.optimizer.state[resumed.model.actor[0].weight]
            for key in ['step','exp_avg','exp_avg_sq']:
                torch.testing.assert_close(original[key],loaded[key],atol=0,rtol=0)
            for label,field,value in [('ratio','tracking_ang_vel',2.1),
                                      ('linear','tracking_lin_vel',5.),
                                      ('other','orientation',-999.)]:
                invalid_env=tensor_env(2);invalid_env.cfg=copy.deepcopy(updated.cfg)
                setattr(invalid_env.cfg.rewards.scales,field,value)
                invalid=SCOPE['TCNPpoRunner'](invalid_env,config,root/label)
                with self.assertRaises(ValueError):invalid.load(checkpoint,allow_yaw_weight_update=True)

    def test_wheel_guidance_resume_requires_explicit_flags_and_keeps_weights(self):
        env=tensor_env(2);config=SCOPE['Q1TCNCfgPPO']()
        env.cfg.rewards.scales.tracking_lin_vel=4.
        env.cfg.rewards.scales.tracking_ang_vel=3.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory);source=SCOPE['TCNPpoRunner'](env,config,root/'source')
            source.iteration=500;source.total_steps=98304000
            checkpoint=source.save()
            saved=SCOPE['load_checkpoint'](checkpoint)
            saved['config']['env']['rewards'].pop('wheel_diff_tracking')
            saved['config']['env']['rewards']['scales'].pop('tracking_wheel_diff')
            torch.save(saved,checkpoint)  # emulate a checkpoint saved before the optional reward existed
            updated=tensor_env(2);updated.cfg=copy.deepcopy(env.cfg)
            updated.cfg.rewards.scales.tracking_ang_vel=2.
            updated.cfg.rewards.scales.tracking_wheel_diff=.5
            updated.cfg.rewards.wheel_diff_tracking=dict(track_width=.299,sigma=1.)
            resumed=SCOPE['TCNPpoRunner'](updated,config,root/'updated')
            with self.assertRaises(ValueError):resumed.load(checkpoint,allow_yaw_weight_update=True)
            with self.assertRaises(ValueError):resumed.load(checkpoint,allow_wheel_diff_update=True)
            resumed.load(checkpoint,allow_yaw_weight_update=True,allow_wheel_diff_update=True)
            self.assertEqual((resumed.iteration,resumed.total_steps),(500,98304000))
            for key,value in source.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)
            updated.cfg.rewards.scales.orientation=-999.
            invalid=SCOPE['TCNPpoRunner'](updated,config,root/'invalid')
            with self.assertRaises(ValueError):
                invalid.load(checkpoint,allow_yaw_weight_update=True,allow_wheel_diff_update=True)


if __name__ == '__main__':
    unittest.main()
