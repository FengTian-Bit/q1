"""From-scratch joint policy, command coverage, resume and public entry tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_q1_flat import ROOT, load_code, make_env
from test_q1_tcn import SCOPE


S = dict(SCOPE)
for filename in ("task_config.py", "joint_task.py", "joint_runner.py"):
    load_code("wheel_legged_gym/envs/q1_tcn/"+filename, S)
SPEC = importlib.util.spec_from_file_location("q1_joint_public_task", ROOT/"wheel_legged_gym/envs/q1_tcn/task.py")
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def args(**overrides):
    values = dict(task="q1_tcn", resume=False, num_envs=None, seed=None,
        max_iterations=None, experiment_name=None, run_name=None,
        load_run=None, checkpoint=None, rl_device="cpu")
    values.update(overrides)
    return SimpleNamespace(**values)


def joint_env(count=8):
    env = make_env(count, task_class=S["Q1TCNJoint"], cfg_class=S["Q1TCNTaskCfg"])
    env.obs_history_length = 16
    env.obs_history = env.obs_buf.repeat(1, 16)
    env.gym.apply_rigid_body_force_tensors = lambda *a: None
    env.get_observations = lambda: (env.obs_buf, env.obs_history)
    env.get_privileged_observations = lambda: env.privileged_obs_buf
    env.check_termination()
    env.compute_observations()
    return env


class JointTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_v7_defaults_and_other_config_objects_remain_independent(self):
        old = S["config_dict"](S["Q1TCNCfg"]())
        cfg, ppo = S["Q1TCNTaskCfg"](), S["Q1TCNTaskCfgPPO"]()
        S["validate_task_config"](cfg, ppo)
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs), (30, 146))
        self.assertEqual(cfg.rewards.reward_version, 7)
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [-1.2, 1.2])
        self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [-1., 1.])
        self.assertEqual(cfg.commands.max_linear_acceleration, 2.)
        self.assertEqual([getattr(cfg.rewards.scales, n) for n in
            ("tracking_lin_vel", "tracking_ang_vel", "tracking_wheel_diff")], [4., 2., .5])
        self.assertEqual(ppo.algorithm.symmetry_loss_coef, .1)
        self.assertEqual(ppo.ablation, "A0")
        self.assertEqual(S["config_dict"](S["Q1TCNCfg"]()), old)
        self.assertEqual(S["Q1TCNCfgPPO"]().runner.experiment_name, "q1_tcn_reward_v2")

    def test_reserved_parking_and_motion_families_and_external_override(self):
        env = joint_env(128)
        env.external_commands = False
        env.commands[:, :2] = .123
        env.episode_length_buf[:] = 300
        env._apply_sequence_commands()
        parked = int(128*.2)
        torch.testing.assert_close(env.commands[:parked, :2], torch.zeros(parked, 2))
        end = parked+int((128-parked)*.75)
        commands = env.commands[parked:end, :2]
        self.assertTrue(((commands[:, 0] == 0) & (commands[:, 1].abs() > .1)).any())
        self.assertTrue(((commands[:, 1] == 0) & (commands[:, 0].abs() > .1)).any())
        self.assertTrue(((commands[:, 0].abs() > .1) & (commands[:, 1].abs() > .1)).any())
        original = env.commands.clone()
        env.commands[:, :2] = .321
        env._apply_sequence_commands(torch.tensor([0, parked]))
        torch.testing.assert_close(env.commands[1, :2], torch.full((2,), .321))
        env.external_commands = True
        env.commands[:] = original
        env._apply_sequence_commands()
        torch.testing.assert_close(env.commands, original)

    def test_both_actor_backbones_receive_policy_gradient(self):
        ppo = S["Q1TCNTaskCfgPPO"]()
        model = S["JointActorCriticTCN"](30, 146, 6, ablation="A0",
            **S["config_dict"](ppo.policy))
        obs = torch.randn(8, 30)*.05
        obs[:, 6:8] = 0
        obs[:4, 29], obs[4:, 29] = 1, 0
        obs[4:, 6] = 1.2
        frames = obs[:, None, :].repeat(1, 16, 1)
        history = frames.flatten(1)
        distribution, estimates = model.distribution(obs, history)
        torch.testing.assert_close(estimates["parking_weight"], torch.tensor([1.]*4+[0.]*4))
        distribution.mean.square().sum().backward()
        for branch in (model, model.parking_policy):
            for module in (branch.input_projection, branch.tcn[0].conv, branch.actor[0]):
                self.assertGreater(float(module.weight.grad.abs().sum()), 0.)
            self.assertIsNone(branch.velocity_head.weight.grad)
        # The historical constructor continues to freeze its parking actor.
        legacy = S["ActorCriticTCN"](30, 146, 6, ablation="A0",
            **S["config_dict"](ppo.policy))
        self.assertFalse(any(p.requires_grad for p in legacy.parking_policy.parameters()))

    def test_real_ppo_updates_both_random_branches_and_roundtrips(self):
        env = joint_env(8)
        cfg = S["Q1TCNTaskCfgPPO"]()
        cfg.runner.num_steps_per_env = 3
        cfg.algorithm.num_learning_epochs, cfg.algorithm.num_mini_batches = 1, 1
        cfg.algorithm.symmetry_batch_size = 8
        env.external_commands = True
        env.commands[:, :2] = 0
        env.command_targets[:, :2] = 0
        env.command_targets[4:, 0] = .3
        env.parking.stationary[:4], env.parking.locked[:4] = True, True
        env.sensor_cache.zero_()
        env.compute_observations()
        # Force a valid parked history while half the batch drives.
        env.obs_buf[:4, 29] = 1.
        env.obs_history = env.obs_buf.repeat(1, 16)
        with tempfile.TemporaryDirectory(dir=ROOT/".test_tmp") as folder:
            runner = S["JointTCNPpoRunner"](env, cfg, Path(folder)/"first")
            before = [b.actor[-1].weight.detach().clone() for b in (runner.model, runner.model.parking_policy)]
            runner.learn(1)
            for old, branch in zip(before, (runner.model, runner.model.parking_policy)):
                self.assertFalse(torch.equal(old, branch.actor[-1].weight))
            checkpoint = runner.save()
            restored = S["JointTCNPpoRunner"](env, cfg, Path(folder)/"resume")
            restored.load(checkpoint)
            self.assertEqual(restored.iteration, 1)
            for key, value in runner.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[key], value)
            self.assertTrue(restored.model.parking_policy.actor[0].weight.requires_grad)

    def test_public_prepare_never_loads_old_weights_without_resume(self):
        rt = SimpleNamespace(env_config=S["Q1TCNTaskCfg"], train_config=S["Q1TCNTaskCfgPPO"],
            version=S["PROFILE_VERSION"], validate=S["validate_task_config"],
            load=Mock(side_effect=AssertionError("Fresh training loaded a model")))
        cfg, train, checkpoint = ENTRY.prepare(args(num_envs=32, seed=7, max_iterations=2), rt)
        self.assertIsNone(checkpoint)
        rt.load.assert_not_called()
        self.assertEqual((cfg.env.num_envs, cfg.seed, train.seed, train.runner.max_iterations), (32, 7, 7, 2))
        with self.assertRaises(ValueError):
            ENTRY.prepare(args(num_envs=1), rt)
        cfg, _, _ = ENTRY.prepare(args(num_envs=1), rt, for_play=True)
        self.assertEqual(cfg.env.num_envs, 1)

    def test_public_dispatch_does_not_enter_other_task_paths(self):
        fake = ModuleType("wheel_legged_gym.envs.q1_tcn.task")
        fake.train_task, fake.play_task = Mock(return_value="trained"), Mock(return_value="played")
        scope = dict(S, __name__="joint_dispatch_test")
        load_code("wheel_legged_gym/scripts/train.py", scope)
        load_code("wheel_legged_gym/scripts/play.py", scope)
        with patch.dict(sys.modules, {fake.__name__: fake}):
            options = args()
            self.assertEqual(scope["train"](options), "trained")
            self.assertEqual(scope["play"](options), "played")
        fake.train_task.assert_called_once_with(options)
        fake.play_task.assert_called_once_with(options)

    def test_resume_requires_joint_profile_and_restores_saved_configuration(self):
        cfg, train = S["Q1TCNTaskCfg"](), S["Q1TCNTaskCfgPPO"]()
        cfg.rewards.scales.parking_drift = -1.25
        saved = dict(task="q1_tcn", profile_version=S["PROFILE_VERSION"],
                     env=S["config_dict"](cfg), train=S["config_dict"](train))
        rt = SimpleNamespace(root=ROOT, env_config=S["Q1TCNTaskCfg"], train_config=S["Q1TCNTaskCfgPPO"],
            version=S["PROFILE_VERSION"], validate=S["validate_task_config"],
            restore=S["restore_config"], load=lambda path: dict(config=saved))
        with tempfile.TemporaryDirectory(dir=ROOT/".test_tmp") as folder:
            path = Path(folder)/"model_10.pt"
            path.write_bytes(b"mock checkpoint")
            rt.get_load_path = lambda *a, **kw: str(path)
            new_cfg, _, resolved = ENTRY.prepare(args(resume=True, load_run=folder, num_envs=16), rt)
            self.assertEqual(new_cfg.rewards.scales.parking_drift, -1.25)
            self.assertEqual(new_cfg.env.num_envs, 16)
            self.assertEqual(resolved, path.resolve())
            saved.pop("profile_version")
            with self.assertRaisesRegex(ValueError, "standalone joint"):
                ENTRY.prepare(args(resume=True, load_run=folder), rt)

    def test_runner_resume_rejects_changed_reward(self):
        env = joint_env(4)
        cfg = S["Q1TCNTaskCfgPPO"]()
        with tempfile.TemporaryDirectory(dir=ROOT/".test_tmp") as folder:
            runner = S["JointTCNPpoRunner"](env, cfg, Path(folder)/"source")
            path = runner.save()
            changed = copy.deepcopy(env.cfg)
            changed.rewards.scales.tracking_ang_vel += 1.
            env.cfg = changed
            resumed = S["JointTCNPpoRunner"](env, cfg, Path(folder)/"new")
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                resumed.load(path)


if __name__ == "__main__":
    unittest.main()
