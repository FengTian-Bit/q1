"""Behavioral and PPO interface checks; tensor Gym stub, no contact simulation."""
import math
import builtins
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from collections import deque
from types import SimpleNamespace
import unittest

import torch
from torch import optim

from test_q1_flat import NS, make_env, load_code


def improved(count=8, stage="low_speed"):
    cfg = NS["Q1Cfg"]()
    NS["configure_q1_stage"](cfg, stage)
    env = make_env(count, task_class=NS["Q1"], cfg_class=lambda: cfg)
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    return env


class Q1ImprovementTests(unittest.TestCase):
    def test_formal_defaults_and_distinct_network_interface(self):
        cfg, train = NS["Q1Cfg"](), NS["Q1CfgPPO"]()
        self.assertEqual((cfg.env.num_envs, train.runner.max_iterations), (4096, 30000))
        self.assertEqual(train.runner.experiment_name, "q1")
        self.assertEqual(train.policy.num_encoder_obs, 155)
        self.assertEqual(NS["Q1FlatCfg"]().env.num_envs, 1024)
        env = improved()
        obs, critic, reward, done, info, history = env.step(torch.zeros(8, 6))
        self.assertEqual((obs.shape[1], critic.shape[1], history.shape[1]), (31, 151, 155))
        self.assertTrue(torch.isfinite(critic).all())

    def test_commands_have_clear_motion_segments_and_slew_limit(self):
        torch.manual_seed(23)
        env = improved(2048)
        env._resample_commands(torch.arange(env.num_envs))
        target = env.command_targets[:, 0]
        stationary = (target == 0).float().mean().item()
        self.assertTrue(0.20 < stationary < 0.30)
        moving = target[target != 0].abs()
        self.assertTrue((moving >= 0.15).all() and (moving <= 0.3).all())
        self.assertTrue((target > 0).any() and (target < 0).any())
        fraction = torch.exp(-target.square()/env.cfg.rewards.tracking_sigma).mean().item()
        self.assertLess(fraction, 0.35)  # stationary policy formerly got 78.5%
        env.commands[:, 0] = 0.3
        env.command_targets[:, 0] = -0.3
        env.episode_length_buf[:] = 1  # avoid the base resampling tick
        env._post_physics_step_callback()
        torch.testing.assert_close(env.commands[:, 0], torch.full_like(target, 0.296))

    def test_parking_height_is_not_yaw_and_reference_does_not_drift(self):
        env = improved()
        env.command_targets[:, :2] = 0
        env.commands[:, :2] = 0
        for _ in range(120):
            env._update_references()
        self.assertTrue(env.parking_active.all())
        self.assertTrue((env.commands[:, 2] == 0.397).all())
        anchor = env.parking_reference.clone()
        env.root_states[:, 0] += 0.05
        env._update_references()
        torch.testing.assert_close(anchor, env.parking_reference)
        self.assertTrue((env.parking_error_body[:, 0] > 0.049).all())
        self.assertTrue((env._reward_parking_drift() > 0).all())
        self.assertTrue((env.compute_proprioception_observations()[:, 27] > 0).all())
        env.command_targets[:, 0] = 0.2
        env._update_references()
        self.assertFalse(env.parking_active.any())
        self.assertEqual(env.parking_error_body.abs().sum().item(), 0.0)

    def test_path_error_is_observable_and_heading_wraps(self):
        env = improved()
        env.command_targets[:, 0] = 0.2
        env._update_references()
        env.root_states[:, 1] += 0.1
        env._update_references()
        self.assertTrue((env.cross_track > 0.099).all())
        self.assertTrue((env.compute_proprioception_observations()[:, 29] > 0).all())
        env.heading_reference[:] = 2*math.pi-0.05
        env._update_references()
        torch.testing.assert_close(env.heading_error, torch.full((8,), 0.05), atol=1e-6, rtol=0)

    def test_actual_posture_not_target_bias_or_wheel_phase(self):
        env = improved()
        before = env._reward_actual_posture()
        env._compute_torques(torch.full((8, 6), 0.4))
        torch.testing.assert_close(before, env._reward_actual_posture())
        env.dof_pos[:, env.wheel_dof_indices] = 20000.0
        torch.testing.assert_close(before, env._reward_actual_posture())
        env.dof_pos[:, env.leg_dof_indices[1]] += 0.3
        self.assertTrue((env._reward_actual_posture() > before).all())
        self.assertTrue((env._reward_leg_symmetry() > 0).all())

    def test_wheel_feedforward_respects_sign_and_actuator_limits(self):
        env = improved()
        env.commands[:, 0] = 0.2
        torque = env._compute_torques(torch.zeros(8, 6))
        expected = 0.2*0.2/env.cfg.asset.wheel_radius
        torch.testing.assert_close(torque[:, env.wheel_dof_indices], torch.full((8, 2), expected))
        torque = env._compute_torques(torch.full((8, 6), 100.0))
        self.assertTrue((torque.abs() <= env.torque_limits).all())
        self.assertTrue((env.velocity_target.abs() <= env.dof_vel_limits).all())

    def test_world_contact_velocity_accounts_for_total_wheel_rotation(self):
        env = improved()
        wheel = env.rigid_body_states[:, env.feet_indices].clone()
        wheel[:, :, 7] = 0.3
        wheel[:, :, 11] = 0.3/env.cfg.asset.wheel_radius
        env.rigid_body_states[:, env.feet_indices] = wheel
        clearance, force, touch, slip = env._sample_wheel_state()
        self.assertTrue((touch == 1).all())
        self.assertLess(slip.abs().max().item(), 1e-6)
        # Relative joint speed is zero: world link angular velocity is essential.
        self.assertEqual(env.dof_vel[:, env.wheel_dof_indices].abs().sum().item(), 0)
        env.rigid_body_states[:, env.feet_indices, 11] = 0.0
        self.assertGreater(env._sample_wheel_state()[3].abs().min().item(), 0.29)
        # A shifted COM changes the reported linear velocity, not the contact
        # point's actual velocity. Verify the COM-to-contact lever arm.
        env.wheel_com_local[:, :, 2] = 0.03
        angular_speed = 0.3/env.cfg.asset.wheel_radius
        env.rigid_body_states[:, env.feet_indices, 11] = angular_speed
        env.rigid_body_states[:, env.feet_indices, 7] = 0.3+angular_speed*0.03
        self.assertLess(env._sample_wheel_state()[3].abs().max().item(), 1e-6)

    def test_substep_contact_duty_detects_brief_unloading(self):
        env = improved()
        counter = [0]
        def simulate(sim):
            env.contact_forces[:, env.feet_indices[0], 2] = 40.0 if counter[0] % 2 == 0 else 0.0
            counter[0] += 1
        env.gym.simulate = simulate
        env.step(torch.zeros(8, 6))
        torch.testing.assert_close(env.wheel_duty[:, 0], torch.full((8,), 0.5))
        torch.testing.assert_close(env.wheel_duty[:, 1], torch.ones(8))
        env.episode_length_buf[:] = 100
        partial = env._reward_rolling_contact().clone()
        env.wheel_duty[:] = 1.0
        self.assertTrue((partial > env._reward_rolling_contact()).all())

    def test_fall_at_timeout_is_failure_and_diagnostics_survive_reset(self):
        env = improved()
        env.root_states[:, 2] = 0.1
        env.episode_length_buf[:] = int(env.max_episode_length)-1
        env.step(torch.zeros(8, 6))
        self.assertTrue(env.physical_failure.all())
        self.assertFalse(env.time_out_buf.any())
        self.assertTrue((env.last_diagnostics[:, 8] < 0.11).all())
        self.assertTrue((env.base_height > 0.39).all())
        self.assertEqual(env.parking_error_body.abs().sum().item(), 0)
        torch.testing.assert_close(env._reward_termination()*env.reward_scales["termination"], torch.full((8,), -2.0))

    def test_steady_parking_penalty_allows_small_wheel_corrections(self):
        env = improved()
        env.parking_weight[:] = 1.0
        env.dof_vel[:, env.wheel_dof_indices] = 0.1
        self.assertEqual(env._reward_parking_wheel_speed().sum().item(), 0)
        env.dof_vel[:, env.wheel_dof_indices] = 3.0
        self.assertTrue((env._reward_parking_wheel_speed() > 0).all())
        env.parking_weight[:] = 0.0
        self.assertEqual(env._reward_parking_wheel_speed().sum().item(), 0)

    def test_real_ppo_resume_saves_completed_iteration_and_encoder_optimizer(self):
        namespace = dict(NS, optim=optim, os=os, time=time, statistics=statistics,
                         deque=deque, VecEnv=object, SummaryWriter=lambda **kw: SimpleNamespace(close=lambda: None))
        for path in ("modules/actor_critic.py", "modules/actor_critic_sequence.py", "storage/rollout_storage.py",
                     "algorithms/ppo.py", "runners/on_policy_runner.py"):
            load_code("wheel_legged_gym/rsl_rl/"+path, namespace)
        train = NS["Q1CfgPPO"]()
        config = NS["class_to_dict"](train)
        config["runner"].update(num_steps_per_env=2, save_interval=1, logger="tensorboard")
        config["policy"].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8], encoder_hidden_dims=[16, 8])
        config["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
        def runner(folder):
            env = improved(stage="turning")
            env.get_observations = lambda: (env.obs_buf, env.obs_history)
            env.get_privileged_observations = lambda: env.privileged_obs_buf
            env.reset = lambda: (env.obs_buf, env.privileged_obs_buf)
            result = namespace["OnPolicyRunner"](env, config, folder, device="cpu")
            result.log = lambda values: None
            return result
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            first = runner(folder)
            first.learn(2)
            checkpoint = Path(folder)/"model_2.pt"
            data = torch.load(checkpoint)
            self.assertEqual(data["iter"], 2)
            self.assertTrue(data["extra_optimizer_state_dict"]["state"])
            second = runner(folder)
            second.load(str(checkpoint))
            self.assertTrue(second.alg.extra_optimizer.state_dict()["state"])
            second.learn(2)
            self.assertEqual(torch.load(Path(folder)/"model_4.pt")["iter"], 4)
            self.assertTrue((Path(folder)/"model_3.pt").is_file())

    def test_playback_writes_diagnostics_and_runs_inference_frames(self):
        env = improved(2)
        train_cfg = NS["Q1CfgPPO"]()
        runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=object()),
                                 get_inference_policy=lambda device: lambda obs, history: (torch.zeros(2, 6), torch.zeros(2, 3)))
        registry = SimpleNamespace(get_cfgs=lambda task: (env.cfg, train_cfg),
                                   make_env=lambda *a, **kw: (env, env.cfg),
                                   make_alg_runner=lambda *a, **kw: (runner, train_cfg))
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            namespace = dict(NS, csv=csv, json=json, datetime=datetime, os=os,
                             WHEEL_LEGGED_GYM_ROOT_DIR=folder, task_registry=registry,
                             export_policy_as_jit=lambda *a: None,
                             range=lambda count: builtins.range(min(count, 4)))
            load_code("wheel_legged_gym/envs/q1/play_q1.py", namespace)
            namespace["play_q1"](SimpleNamespace(num_envs=2, q1_stage="low_speed", load_run="test", checkpoint=2, play_mode="auto", play_duration=0.04, headless=True))
            output = Path(folder)/"logs/q1/exported/evaluations"
            with next(output.glob("*.csv")).open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(rows[0]), len(NS["DIAGNOSTIC_FIELDS"])+3)
            self.assertTrue(all(len(row) == len(rows[0]) for row in rows))
            summary = json.loads(next(output.glob("*.json")).read_text())
            self.assertEqual(summary["frames"], 4)
            self.assertEqual(summary["env0_resets"], 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
