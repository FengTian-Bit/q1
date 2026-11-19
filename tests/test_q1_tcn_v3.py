"""V3 configuration, real network/PPO, and checkpoint guards on CPU.

Tensor-only environment fixtures do not validate physical stair performance.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
import sys
from types import ModuleType

import torch

from test_q1_flat import load_code, make_env
from test_q1_tcn import SCOPE


PROFILE = Path(__file__).resolve().parents[1]/'wheel_legged_gym/envs/q1_tcn/v3_profile.py'
V3 = dict(SCOPE, __file__=str(PROFILE))
load_code('wheel_legged_gym/envs/q1_tcn/v3_profile.py', V3)


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Initialize lazy optimizer imports before patch.dict(sys.modules),
        # which restores the module table after the public-entry fixture.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]).step()

    def test_profile_exact_and_independent_instances(self):
        cfg, train = V3['Q1TCNTaskCfg'](), V3['Q1TCNTaskCfgPPO']()
        V3['validate_task_config'](cfg, train)
        expected = V3['profile_config']()
        self.assertEqual(V3['config_dict'](cfg), expected['env'])
        self.assertEqual(V3['config_dict'](train), expected['train'])
        self.assertEqual(cfg.env.obs_history_length, 61)
        self.assertEqual(cfg.control.decimation*cfg.sim.dt, .01)
        self.assertEqual(cfg.control.leg_action_scales, [.5, 1.5, .5, 1.5])
        self.assertEqual(cfg.rewards.scales.parking_drift, -.3)
        self.assertEqual(train.algorithm.entropy_coef, .002)
        self.assertEqual(train.algorithm.desired_kl, .004)
        cfg.terrain.num_rows = 1
        self.assertEqual(V3['Q1TCNTaskCfg']().terrain.num_rows, 20)

    def test_full_network_and_slow_history(self):
        cfg, train = V3['Q1TCNTaskCfg'](), V3['Q1TCNTaskCfgPPO']()
        model = V3['ActorCriticTCN'](30, 146, 6, ablation=train.ablation,
                                    **V3['config_dict'](train.policy))
        obs = torch.randn(3, 30)
        history = torch.randn(3, 1830)
        action, estimates = model.act_inference(obs, history)
        self.assertEqual(action.shape, (3, 6))
        self.assertEqual(estimates['slow_latent'].shape, (3, 16))
        self.assertEqual(estimates['slope'].shape, (3, 2))
        self.assertEqual(estimates['clearance'].shape, (3, 2))
        changed = history.clone()
        changed[:, :-480] += .7
        self.assertFalse(torch.equal(action, model.act_inference(obs, changed)[0]))
        self.assertTrue((action[:, [0, 3]].abs() <= .7).all())
        self.assertTrue((action[:, [2, 5]].abs() <= 2.85).all())

    def test_v3_real_rollout_update_and_restore(self):
        # Use the production profile on a plane fixture, retaining61 history,
        # teacher clock, odometry, enabled terrain heads and real PPO update.
        env = make_env(8, task_class=V3['Q1TCN'], cfg_class=V3['Q1TCNTaskCfg'])
        env.obs_history_length = 61
        env.obs_history = env.obs_buf.repeat(1, 61)
        env.gym.apply_rigid_body_force_tensors = lambda *args: None
        env.get_observations = lambda: (env.obs_buf, env.obs_history)
        env.get_privileged_observations = lambda: env.privileged_obs_buf
        env.set_training_iteration(11000)
        env.reset_idx(torch.arange(env.num_envs))
        env.check_termination()
        env.compute_observations()
        self.assertEqual(float(env.ctbc_feedforward_weight), 0.)
        train = V3['Q1TCNTaskCfgPPO']()
        train.runner.num_steps_per_env = 2
        train.algorithm.num_learning_epochs = 1
        train.algorithm.num_mini_batches = 1
        with tempfile.TemporaryDirectory() as folder:
            runner = V3['TCNPpoRunner'](env, train, Path(folder)/'first')
            before = runner.model.actor[0].weight.detach().clone()
            runner.learn(1)
            self.assertEqual(runner.iteration, 1)
            self.assertTrue(runner.alg.optimizer.state)
            self.assertFalse(torch.equal(before, runner.model.actor[0].weight))
            restored = V3['TCNPpoRunner'](env, train, Path(folder)/'restored')
            restored.load(runner.save())
            self.assertEqual(restored.iteration, runner.iteration)
            self.assertEqual(restored.total_steps, runner.total_steps)
            for key, value in runner.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            train.policy.slow_channels = 64
            wrong = V3['TCNPpoRunner'](env, train, Path(folder)/'wrong')
            with self.assertRaisesRegex(ValueError, 'Resume configuration mismatch'):
                wrong.load(runner.save())

    def test_public_training_entry_uses_v3_and_absolute_budget(self):
        package = ModuleType('_q1_v3_public_test')
        package.__path__ = [str(PROFILE.parent)]
        modules = {package.__name__: package}
        for name in ['q1_tcn_config', 'v3_profile', 'runner']:
            module = ModuleType(package.__name__+'.'+name)
            module.__dict__.update(V3)
            modules[package.__name__+'.'+name] = module
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependency = root/'wheel_legged_gym/envs/q1_flat/q1_flat.py'
            dependency.parent.mkdir(parents=True)
            dependency.write_text('# source snapshot fixture')
            gym = ModuleType('wheel_legged_gym')
            gym.WHEEL_LEGGED_GYM_ROOT_DIR = str(root)
            utils = ModuleType('wheel_legged_gym.utils')
            def create_env(name, args, env_cfg):
                self.assertEqual(name, 'q1_tcn')
                env = make_env(4, task_class=V3['Q1TCN'], cfg_class=lambda: env_cfg)
                env.obs_history_length = 61
                env.obs_history = env.obs_buf.repeat(1, 61)
                env.gym.apply_rigid_body_force_tensors = lambda *a: None
                env.gym.destroy_sim = lambda *a: None
                env.get_observations = lambda: (env.obs_buf, env.obs_history)
                env.get_privileged_observations = lambda: env.privileged_obs_buf
                def reset():
                    env.reset_idx(torch.arange(4))
                    env.check_termination()
                    env.compute_observations()
                env.reset = reset
                return env, env_cfg
            utils.task_registry = SimpleNamespace(make_env=create_env)
            modules.update(wheel_legged_gym=gym)
            modules['wheel_legged_gym.utils'] = utils
            path = PROFILE.with_name('v3_task.py')
            spec = importlib.util.spec_from_file_location(package.__name__+'.v3_task', path)
            task = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(task)
                args = SimpleNamespace(seed=None, num_envs=4, max_iterations=1, resume=False,
                                       experiment_name=None, run_name=None, rl_device='cpu')
                output = task.train_task(args)
            saved = torch.load(output/'model_1.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['iteration'], 1)
            self.assertEqual(saved['config']['env']['env']['obs_history_length'], 61)
            self.assertTrue(saved['config']['train']['policy']['slow_branch'])
            manifest = json.loads((output/'initialization.json').read_text())
            self.assertEqual(manifest['initialization'], 'random')
            self.assertEqual(manifest['initial_iteration'], 0)
            self.assertFalse(manifest['policy_and_adam_inherited'])


if __name__ == '__main__':
    unittest.main()
