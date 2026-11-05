"""CPU numerical tests. Tensor Gym stubs do not establish physical balance."""

import csv
import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from test_q1_flat import NS, load_code, make_env, TensorGym


LOCAL = "wheel_legged_gym/envs/q1_tcn/"
SCOPE = dict(NS, F=F, Normal=Normal, Path=Path, csv=csv, json=json,
             gymapi=SimpleNamespace(ENV_SPACE=0), argparse=argparse, random=random,
             deepcopy=copy.deepcopy, copy=copy, sys=sys)
for filename in ("q1_tcn_config.py", "labels.py", "parking.py", "odometry.py", "curriculum.py", "command_reference.py", "actor_critic.py", "symmetry.py", "ppo.py", "q1_tcn.py",
                 "metrics.py", "runner.py", "cli.py", "evaluate.py"):
    # Avoid the evaluate.py __main__ entry and imports of unavailable Isaac Gym.
    SCOPE["__name__"] = "q1_tcn_cpu_tests"
    load_code(LOCAL+filename, SCOPE)

Model = SCOPE["ActorCriticTCN"]
Config, TrainConfig = SCOPE["Q1TCNCfg"], SCOPE["Q1TCNCfgPPO"]
PPO = SCOPE["PPO"]
Task = SCOPE["Q1TCN"]


def tensor_env(count=4):
    env = make_env(count, task_class=Task, cfg_class=Config)
    env.obs_history_length = 16
    env.obs_history = env.obs_buf.repeat(1, 16)
    env.gym.apply_rigid_body_force_tensors = lambda *args: None
    env.get_observations = lambda: (env.obs_buf, env.obs_history)
    env.get_privileged_observations = lambda: env.privileged_obs_buf
    env.check_termination()
    env.compute_observations()
    return env


class TCNNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_capacity_masks_and_ppo_gradient(self):
        counts = []
        for name, enabled in SCOPE["ABLATIONS"].items():
            torch.manual_seed(1)
            model = Model(27, 143, 6, ablation=name)
            counts.append(sum(p.numel() for p in model.parameters()))
            obs, history = torch.randn(3, 27), torch.randn(3, 432)
            dist, estimates = model.distribution(obs, history)
            self.assertEqual(dist.mean.shape, (3, 6))
            self.assertEqual(model.receptive_field, 16)
            dist.mean.sum().backward()
            self.assertGreater(model.input_projection.weight.grad.abs().sum(), 0)
            for head in (model.velocity_head, model.contact_head, model.slip_head):
                self.assertIsNone(head.weight.grad)  # explicit outputs are detached
            captured = []
            hook = model.actor[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
            model.distribution(obs, history)
            hook.remove()
            for value, active in zip(captured[0][:, -7:].split([3, 2, 2], dim=-1), enabled):
                if not active:
                    self.assertEqual(float(value.detach().abs().sum()), 0)
        self.assertEqual(len(set(counts)), 1)

    def test_causality_and_window_validation(self):
        model = Model(27, 143, 6)
        x = torch.randn(2, 64, 16)
        changed = x.clone()
        changed[:, :, 8:] += 100
        torch.testing.assert_close(model.tcn(x)[:, :, :8], model.tcn(changed)[:, :, :8])
        with self.assertRaises(ValueError):
            Model(27, 143, 6, history_length=32)

    def test_contact_debounce_and_airborne_slip_loss(self):
        update = SCOPE["update_contact"]
        contact = torch.zeros(1, 2, dtype=torch.bool)
        counter = torch.zeros(1, 2, dtype=torch.long)
        force = torch.tensor([[10., 0.]])
        contact, counter = update(force, contact, counter, 5, 3, 2)
        self.assertFalse(contact.any())
        contact, counter = update(force, contact, counter, 5, 3, 2)
        self.assertTrue(contact[0, 0])
        model = Model(27, 143, 6, ablation="A4")
        estimates = model.encode(torch.randn(2, 432))
        targets = torch.zeros(2, 10)
        targets[:, 7] = 1
        targets[:, 3:5] = 1  # debounce can still report contact after takeoff
        losses = SCOPE["estimation_losses"](estimates, targets, model.enabled)
        self.assertEqual(float(losses["slip"].detach()), 0)
        targets[:, 8] = 1
        losses = SCOPE["estimation_losses"](estimates, targets, model.enabled)
        self.assertGreater(float(losses["slip"].detach()), 0)
        targets[:, 7] = 0
        losses = SCOPE["estimation_losses"](estimates, targets, model.enabled)
        self.assertEqual(sum(float(value.detach()) for value in losses.values()), 0)

    def test_slip_absolute_rotation_and_com_offset(self):
        states = torch.zeros(1, 2, 13)
        states[:, :, 6] = 1
        states[:, :, 7] = 1  # COM vx
        states[:, :, 11] = 2  # absolute angular velocity about +Y
        centers = torch.zeros(1, 2, 3)
        com = torch.zeros_like(centers)
        com[:, :, 2] = 0.01
        q = torch.tensor([[0., 0., 0., 1.]])
        slip = SCOPE["flat_wheel_slip"](states, centers, com, 0.1, 0.03, q)
        torch.testing.assert_close(slip, torch.full((1, 2), 0.78))

    def test_environment_alignment_reset_and_terminal_state(self):
        env = tensor_env()
        env.contact_forces[:, env.feet_indices, 2] = 10
        env.step(torch.zeros(4, 6))
        env.step(torch.zeros(4, 6))
        self.assertEqual(env.obs_history.shape, (4, 432))
        self.assertEqual(env.privileged_obs_buf.shape, (4, 143))
        self.assertTrue(env.estimator_targets[:, 3:5].bool().all())
        torch.testing.assert_close(env.estimator_targets[:, :3], env.base_lin_vel)
        env.root_states[0, 2] = 0.1
        env.episode_length_buf[0] = env.max_episode_length
        env.step(torch.zeros(4, 6))
        self.assertTrue(env.transition_metrics["failure"][0])
        self.assertAlmostEqual(float(env.transition_metrics["position"][0, 2]), 0.1, places=5)
        self.assertEqual(float(env.estimator_targets[0].abs().sum()), 0)
        history = env.obs_history[0].reshape(16, 27)
        torch.testing.assert_close(history, env.obs_buf[0].expand(16, -1))

    def test_force_pulse_and_zero_force_release(self):
        env = tensor_env()
        applied = []
        env.gym.apply_rigid_body_force_tensors = lambda sim, force, torque, space: applied.append(force.clone())
        env.envs_steps_buf[:] = 1001  # 5.005 s
        env._push_robots()
        self.assertGreater(float(applied[-1].abs().sum()), 0)
        held = applied[-1].clone()
        env.envs_steps_buf[:] = 1010
        env._push_robots()
        torch.testing.assert_close(applied[-1], held)
        env.envs_steps_buf[:] = 1031
        env._push_robots()
        self.assertEqual(float(applied[-1].abs().sum()), 0)

    def test_gae_terminal_boundaries(self):
        returns, _ = SCOPE["compute_gae"](torch.tensor([[1.], [100.]]),
            torch.tensor([[True], [False]]), torch.zeros(2, 1), torch.zeros(1), 0.99, 0.95)
        self.assertEqual(float(returns[0]), 1)

    def test_timeout_uses_terminal_value_and_failure_never_bootstraps(self):
        env = tensor_env()
        cfg = TrainConfig()
        cfg.runner.num_steps_per_env = 1
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            runner = SCOPE["TCNPpoRunner"](env, cfg, folder)
            runner.model.evaluate = lambda obs: obs[:, 0]
            env.root_states[:, 7] = 0.4
            env.episode_length_buf[:] = env.max_episode_length-1
            env.contact_forces[0, 0, 2] = 10  # physical failure and timeout coincide
            batch, _, failures, timeouts = runner.collect()
            delta = batch["rewards"][0]-batch["environment_rewards"][0]
            self.assertEqual(float(delta[0]), 0)
            self.assertEqual(failures, 1)
            self.assertEqual(timeouts, 3)
            torch.testing.assert_close(delta[1:], torch.full((3,), 0.4*env.obs_scales.lin_vel*runner.alg.gamma))

    def test_real_rollout_update_all_ablations(self):
        for name in SCOPE["ABLATIONS"]:
            env = tensor_env()
            cfg = TrainConfig()
            cfg.ablation = name
            cfg.runner.num_steps_per_env = 3
            cfg.algorithm.num_learning_epochs = 1
            cfg.algorithm.num_mini_batches = 2
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
                runner = SCOPE["TCNPpoRunner"](env, cfg, folder)
                batch, last, _, _ = runner.collect()
                # Stored targets belong to the observation before env.step.
                torch.testing.assert_close(batch["targets"][0, :, :3], batch["critic_obs"][0, :, :3] / env.obs_scales.lin_vel)
                before = runner.model.input_projection.weight.detach().clone()
                metrics = runner.alg.update(batch, last)
                self.assertTrue(all(np.isfinite(list(metrics.values()))))
                self.assertFalse(torch.equal(before, runner.model.input_projection.weight))
                if name == "A0":
                    self.assertEqual(metrics["velocity_loss"], 0)
                    self.assertEqual(metrics["contact_loss"], 0)
                    self.assertEqual(metrics["slip_loss"], 0)
                # Round-trip a real checkpoint, including runtime-derived config.
                checkpoint = runner.save()
                other_cfg, other_env = TrainConfig(), tensor_env()
                other_cfg.ablation = name
                other_cfg.runner.num_steps_per_env = 3
                other_cfg.algorithm.num_learning_epochs = 1
                other_cfg.algorithm.num_mini_batches = 2
                other = SCOPE["TCNPpoRunner"](other_env, other_cfg, Path(folder)/"restored")
                other.load(checkpoint)
                for left, right in zip(runner.model.parameters(), other.model.parameters()):
                    torch.testing.assert_close(left, right)

    def test_failed_trials_are_not_reintroduced_and_disabled_heads_are_na(self):
        acc = SCOPE["TrialMetrics"](2, 0.01, np.zeros((2, 3)), (False, False, False))
        sample = dict(velocity=np.zeros((2, 3)), command=np.zeros(2), pitch=np.zeros(2), roll=np.zeros(2),
            support=np.ones((2, 2), dtype=bool), slip=np.zeros((2, 2)), saturation=np.zeros(2),
            position=np.zeros((2, 3)), force=np.zeros((2, 2)), failure=np.array([True, False]))
        acc.update(0.01, sample)
        sample["position"][0, 0] = 100
        sample["failure"][:] = False
        acc.update(0.02, sample)
        rows = acc.rows(101, "stand")
        self.assertEqual(rows[0]["observed_s"], 0.01)
        self.assertEqual(rows[0]["max_displacement_m"], 0)
        summary = SCOPE["summarize"](rows)
        self.assertEqual(summary["stand"]["metrics"]["failed"]["mean"], 0.5)
        self.assertIsNone(summary["stand"]["metrics"]["contact_f1"]["mean"])
        json.dumps(summary, allow_nan=False)

    def test_saved_config_and_evaluation_scenarios(self):
        # Include fields materialized by LeggedRobot._parse_cfg.
        saved = dict(env=SCOPE["config_dict"](tensor_env().cfg), train=SCOPE["config_dict"](TrainConfig()))
        cfg, train = SCOPE["scenario_config"](saved, "payload", 101, 8, 12)
        self.assertEqual(cfg.domain_rand.added_mass_range, [0.75, 0.75])
        self.assertEqual(cfg.estimator.payload_com_offset_x, 0.02)
        cfg, train = SCOPE["scenario_config"](saved, "low_friction", 101, 8, 12)
        self.assertEqual(cfg.terrain.static_friction, 0.2)
        self.assertEqual(Config().terrain.static_friction, 0.8)

    def test_external_evaluation_initializes_height_before_reset_history(self):
        env = tensor_env()
        env.commands.zero_()  # same initial commands as a freshly constructed task
        env.external_commands = True
        env.evaluation_mode = True
        env._resample_commands(torch.arange(env.num_envs))
        self.assertEqual(float(env.commands[:, 2].sum()), 0)
        SCOPE["set_evaluation_commands"](env, 0.0)
        env.reset_idx(torch.arange(env.num_envs))
        env.compute_observations()
        expected = torch.full((env.num_envs,), 0.397)
        torch.testing.assert_close(env.commands[:, 2], expected)
        torch.testing.assert_close(env.obs_buf[:, 8], expected*env.commands_scale[2])
        history = env.obs_history.reshape(env.num_envs, 16, 27)
        torch.testing.assert_close(history[:, :, 8], (expected*env.commands_scale[2])[:, None].expand(-1, 16))
        SCOPE["set_evaluation_commands"](env, -0.25)
        torch.testing.assert_close(env.commands[:, 2], expected)

    def test_cli_list_options_and_comparison(self):
        parser = SCOPE["task_parser"](evaluation=True)
        args, remaining = parser.parse_known_args(["--checkpoint_path", "model.pt", "--eval_seeds", "101", "102",
            "--scenarios", "stand", "tracking", "--headless", "--sim_device", "cuda:0"])
        self.assertEqual(args.eval_seeds, [101, 102])
        self.assertEqual(args.scenarios, ["stand", "tracking"])
        self.assertEqual(remaining, ["--headless", "--sim_device", "cuda:0"])
        namespace = dict(SCOPE)
        load_code(LOCAL+"compare.py", namespace)
        report = dict(ablation="A1", training_seed=1, training_steps=100,
            training_protocol=dict(env=dict(seed=1)), evaluation=dict(duration=12),
            scenarios=dict(stand=dict(metrics=dict(failed=dict(mean=0.2)))))
        other = copy.deepcopy(report)
        other["training_seed"] = 2
        other["training_protocol"]["env"]["seed"] = 2
        other["scenarios"]["stand"]["metrics"]["failed"]["mean"] = 0.4
        rows = namespace["comparison_rows"]([report, other])
        self.assertAlmostEqual(rows[0]["mean"], 0.3)
        self.assertAlmostEqual(rows[0]["std_across_training_seeds"], np.sqrt(0.02))
        with self.assertRaises(ValueError):
            namespace["comparison_rows"]([report, report])
        other["training_steps"] = 101
        with self.assertRaises(ValueError):
            namespace["comparison_rows"]([report, other])

    def test_plot_and_csv_artifacts(self):
        acc = SCOPE["TrialMetrics"](2, 0.01, np.zeros((2, 3)), (False, False, False))
        sample = dict(velocity=np.zeros((2, 3)), command=np.zeros(2), pitch=np.zeros(2), roll=np.zeros(2),
            support=np.ones((2, 2), dtype=bool), slip=np.zeros((2, 2)), saturation=np.zeros(2),
            position=np.zeros((2, 3)), force=np.zeros((2, 2)), failure=np.zeros(2, dtype=bool))
        for i in range(10):
            acc.update((i+1)*0.01, sample)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            plot = Path(folder)/"response.png"
            SCOPE["plot_trace"](plot, [(101, acc.trace)], "Synthetic plotting test")
            self.assertGreater(plot.stat().st_size, 1000)
            output = Path(folder)/"trials.csv"
            SCOPE["write_csv"](output, acc.rows(101, "stand"))
            with output.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
