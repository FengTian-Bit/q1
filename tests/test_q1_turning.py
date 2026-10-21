"""Turning sampling, differential drive, reference transitions and reward timing."""
import math
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

from test_q1 import improved
from test_q1_flat import NS, ROOT, load_code


class Q1TurningTests(unittest.TestCase):
    def test_balanced_stand_straight_spin_arc_samples(self):
        torch.manual_seed(120)
        env = improved(12000, stage="turning")
        env._resample_commands(torch.arange(env.num_envs))
        vx, yaw = env.command_targets[:, 0], env.command_targets[:, 1]
        categories = [(vx == 0) & (yaw == 0), (vx != 0) & (yaw == 0),
                      (vx == 0) & (yaw != 0), (vx != 0) & (yaw != 0)]
        for mask in categories:
            self.assertTrue(0.23 < mask.float().mean().item() < 0.27)
        for mask in categories[2:]:
            fraction = (yaw[mask] > 0).float().mean().item()
            self.assertTrue(0.46 < fraction < 0.54)
        arc = categories[3]
        for sign_v in (-1, 1):
            for sign_y in (-1, 1):
                self.assertTrue(((vx[arc]*sign_v > 0) & (yaw[arc]*sign_y > 0)).any())
        self.assertTrue((yaw[yaw != 0].abs() >= 0.2).all())
        self.assertTrue((yaw.abs() <= 0.6).all())
        self.assertTrue((vx.abs() <= 0.8).all())
        untouched = env.command_targets[1:].clone()
        env._resample_commands(torch.tensor([0]))
        torch.testing.assert_close(env.command_targets[1:], untouched)

    def test_yaw_slew_reversal_and_parking_waits_for_yaw_braking(self):
        env = improved(2, stage="turning")
        env.external_commands = True
        env.command_targets[:, :2] = 0
        env.commands[:, :2] = 0
        env.commands[:, 1] = 0.6
        env.command_targets[:, 1] = -0.6
        env.episode_length_buf[:] = 1
        env._post_physics_step_callback()
        torch.testing.assert_close(env.commands[:, 1], torch.full((2,), 0.592))
        torch.testing.assert_close(env.yaw_acceleration, torch.full((2,), -0.8), atol=1e-5, rtol=0)
        env.command_targets[:, 1] = 0
        env.idle_elapsed[:] = 10
        env._update_references()
        self.assertFalse(env.parking_active.any())
        self.assertEqual(env.idle_elapsed.sum().item(), 0)
        env.commands[:, 1] = 0
        for _ in range(120):
            env._update_references()
        self.assertTrue(env.parking_active.all())

    def test_turns_release_straight_reference_then_reanchor_after_ramp(self):
        env = improved(2, stage="turning")
        env.command_targets[:, :2] = torch.tensor([0.2, 0.0])
        env.commands[:, :2] = torch.tensor([0.2, 0.0])
        env._update_references()
        self.assertTrue(env.path_active.all())
        env.command_targets[:, 1] = 0.6
        env.root_states[:, 1] += 0.3
        yaw = math.pi/2
        env.base_quat[:, :] = torch.tensor([0, 0, math.sin(yaw/2), math.cos(yaw/2)])
        env._update_references()
        self.assertFalse(env.path_active.any())
        self.assertFalse(env.parking_active.any())
        self.assertEqual(env._reward_straight_path().sum().item(), 0)
        self.assertEqual(env.compute_proprioception_observations()[:, 27:].abs().sum().item(), 0)
        env.command_targets[:, 1] = 0
        env.commands[:, 1] = 0.2
        env._update_references()
        self.assertFalse(env.path_active.any())
        env.commands[:, 1] = 0
        env._update_references()
        self.assertTrue(env.path_active.all())
        torch.testing.assert_close(env.path_reference, env.root_states[:, :2])
        self.assertLess(env.heading_error.abs().max().item(), 1e-6)
        self.assertLess(env.cross_track.abs().max().item(), 1e-6)

    def test_differential_feedforward_sign_limits_and_urdf_geometry(self):
        env = improved(4, stage="turning")
        env.commands[:, :2] = torch.tensor([[0, 0.6], [0, -0.6], [0.3, 0.6], [-0.3, -0.6]])
        torque = env._compute_torques(torch.zeros(4, 6))
        expected = (env.commands[:, :1]-env.commands[:, 1:2]*env.wheel_lateral_offsets)/env.cfg.asset.wheel_radius
        torch.testing.assert_close(env.velocity_target[:, env.wheel_dof_indices], expected)
        self.assertLess(torque[0, env.wheel_dof_indices[0]].item(), 0)
        self.assertGreater(torque[0, env.wheel_dof_indices[1]].item(), 0)
        self.assertTrue((torque.abs() <= env.torque_limits).all())
        env._compute_torques(torch.full((4, 6), 100.0))
        self.assertTrue((env.velocity_target.abs() <= env.dof_vel_limits).all())
        root = ET.parse(ROOT/"resources/robots/q1/urdf/q1.urdf").getroot()
        by_child = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
        for index, side in enumerate(("L", "R")):
            link, y = side+"_foot", env.cfg.rewards.wheel_center_y[index]
            while link != "trunk":
                joint = by_child[link]
                origin = joint.find("origin").attrib
                self.assertEqual(origin.get("rpy", "0 0 0"), "0 0 0")
                y += float(origin["xyz"].split()[1])
                link = joint.find("parent").attrib["link"]
            self.assertAlmostEqual(y, env.cfg.asset.wheel_lateral_offsets[index], places=6)

    def test_yaw_reward_uses_command_applied_during_physics(self):
        env = improved(2, stage="turning")
        env.applied_commands[:, :2] = torch.tensor([[0.0, 0.6], [0.2, -0.6]])
        env.commands[:, :2] = 0  # next command differs from the physics interval
        env.base_ang_vel[:, 2] = torch.tensor([0.6, -0.6])
        torch.testing.assert_close(env._reward_tracking_ang_vel(), torch.ones(2))
        torch.testing.assert_close(env._reward_turning_yaw_tracking(), torch.tensor([1.0, 0.0]))
        env.base_ang_vel[:, 2] = 0
        self.assertLess(env._reward_tracking_ang_vel().max().item(), 0.001)
        self.assertTrue((env._reward_turning_yaw_tracking() >= 0).all())

    def test_turning_diagnostics_survive_failure_and_episode_reports_yaw(self):
        env = improved(2, stage="turning")
        env.set_command_target(0.0, 0.6)
        env.commands[:, 1] = 0.2
        env.root_states[:, 2] = 0.1
        env.step(torch.zeros(2, 6))
        fields = NS["DIAGNOSTIC_FIELDS"]
        torch.testing.assert_close(env.last_diagnostics[:, fields.index("target_yaw_rate")], torch.full((2,), 0.6))
        torch.testing.assert_close(env.last_diagnostics[:, fields.index("applied_yaw_command")], torch.full((2,), 0.2))
        self.assertTrue((env.last_diagnostics[:, fields.index("height")] < 0.11).all())
        self.assertIn("q1_yaw_mae", env.extras["episode"])
        self.assertEqual(env.yaw_acceleration.abs().sum().item(), 0)

    def test_stage_configuration_and_default_training_entrypoint(self):
        cfg = NS["Q1Cfg"]()
        self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [-0.6, 0.6])
        for stage in ("balance", "low_speed", "turning"):
            NS["configure_q1_stage"](cfg, stage)
            self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [-0.6, 0.6] if stage == "turning" else [0.0, 0.0])
        config_module = ModuleType("wheel_legged_gym.envs.q1.q1_config")
        config_module.configure_q1_stage = NS["configure_q1_stage"]

        class ReadyForGym(Exception):
            pass

        def make_env(**kwargs):
            raise ReadyForGym()

        for stage in (None, "balance", "low_speed", "turning"):
            cfg, train = NS["Q1Cfg"](), NS["Q1CfgPPO"]()
            registry = SimpleNamespace(get_cfgs=lambda task: (cfg, train), make_env=make_env)
            train_ns = dict(NS, task_registry=registry)
            load_code("wheel_legged_gym/scripts/train.py", train_ns)
            args = SimpleNamespace(task="q1", q1_stage=stage, run_name=None)
            with patch.dict(sys.modules, {config_module.__name__: config_module}):
                with self.assertRaises(ReadyForGym):
                    train_ns["train"](args)
            self.assertEqual(args.q1_stage, stage or "turning")
            self.assertEqual(train.runner.run_name, stage or "turning")
            self.assertEqual((cfg.env.num_envs, train.runner.max_iterations), (4096, 30000))

    def test_headless_default_auto_evaluates_both_turn_directions_and_arcs(self):
        env = improved(2, stage="turning")
        env.dt = 0.5  # Accelerate the stub's evaluation clock, preserving all segments.
        train = NS["Q1CfgPPO"]()
        runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=object()),
                                 get_inference_policy=lambda device: lambda obs, history: (torch.zeros(2, 6), None))
        registry = SimpleNamespace(get_cfgs=lambda task: (env.cfg, train),
                                   make_env=lambda *a, **kw: (env, env.cfg),
                                   make_alg_runner=lambda *a, **kw: (runner, train))
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            play_ns = dict(NS, csv=csv, datetime=datetime, json=json, os=os, task_registry=registry,
                           export_policy_as_jit=lambda *a: None, WHEEL_LEGGED_GYM_ROOT_DIR=folder)
            load_code("wheel_legged_gym/envs/q1/play_q1.py", play_ns)
            args = SimpleNamespace(num_envs=2, q1_stage=None, headless=True, play_mode="keyboard",
                                   play_duration=0.0, load_run="test", checkpoint=1)
            play_ns["play_q1"](args)
            output = Path(folder)/"logs/q1/exported/evaluations"
            summary = json.loads(next(output.glob("*.json")).read_text())
            self.assertEqual((summary["stage"], summary["play_mode"], summary["frames"]), ("turning", "auto", 144))
            with next(output.glob("*.csv")).open() as stream:
                rows = list(csv.DictReader(stream))
            commands = {(float(row["target_vx"]), float(row["target_yaw_rate"])) for row in rows}
            self.assertTrue(any(v == 0 and y > 0.5 for v, y in commands))
            self.assertTrue(any(v == 0 and y < -0.5 for v, y in commands))
            self.assertTrue(any(v > 0 and y > 0 for v, y in commands))
            self.assertTrue(any(v < 0 and y < 0 for v, y in commands))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
