"""Exercise actual keyboard, viewer dispatch and playback with a tensor Gym stub."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from test_q1 import improved
from test_q1_flat import NS, load_code


namespace = dict(NS)
load_code("wheel_legged_gym/envs/q1/keyboard.py", namespace)
Q1Keyboard = namespace["Q1Keyboard"]
KEYS = list(Q1Keyboard.bindings) + ["KEY_"+str(n) for n in range(9)] + ["KEY_R"]
GYMAPI = SimpleNamespace(**{key: key for key in KEYS})


def event(action, value=1):
    return SimpleNamespace(action=action, value=value)


def viewer_env(count=3):
    env = improved(count)
    env.viewer = object()
    env.subscriptions, env.cameras = [], []
    env.gym.subscribe_viewer_keyboard_event = lambda viewer, key, action: env.subscriptions.append((key, action))
    env.set_camera = lambda position, target: env.cameras.append((position, target))
    return env


class Q1KeyboardTests(unittest.TestCase):
    def test_press_release_limit_and_selected_robot_commands(self):
        env = viewer_env()
        keys = Q1Keyboard(env)
        keys.enable(GYMAPI)
        self.assertEqual(len(env.subscriptions), len(KEYS))
        keys.handle_event(event("q1_vx_plus"))
        keys.handle_event(event("q1_vx_plus", 0))
        self.assertAlmostEqual(keys.speed[0].item(), 0.2)
        keys.handle_event(event("q1_vx_plus"))
        self.assertAlmostEqual(keys.speed[0].item(), 0.4)
        for _ in range(4):
            keys.handle_event(event("q1_vx_plus"))
        self.assertAlmostEqual(keys.speed[0].item(), 0.8)
        keys.handle_event(event("q1_next"))
        for _ in range(8):
            keys.handle_event(event("q1_vx_minus"))
        keys.apply()
        torch.testing.assert_close(env.command_targets[:, 0], torch.tensor([0.8, -0.8, 0.0]))
        self.assertTrue((env.command_targets[:, 1] == 0).all())
        self.assertTrue((env.command_targets[:, 2] == env.cfg.commands.ranges.height[0]).all())
        keys.handle_event(event("q1_stop"))
        keys.apply()
        torch.testing.assert_close(env.command_targets[:, 0], torch.tensor([0.8, 0.0, 0.0]))
        keys.close()

    def test_selection_bounds_wrapping_and_turning_commands(self):
        env = viewer_env()
        keys = Q1Keyboard(env)
        keys.handle_event(event("q1_prev"))
        self.assertEqual(keys.selected, 2)
        keys.handle_event(event("q1_next"))
        self.assertEqual(keys.selected, 0)
        keys.handle_event(event("q1_select_1"))
        keys.handle_event(event("q1_select_8"))
        self.assertEqual(keys.selected, 1)
        keys.handle_event(event("q1_vx_plus"))
        before = keys.speed.clone()
        keys.handle_event(event("q1_yaw_plus"))
        keys.handle_event(event("q1_yaw_plus", 0))
        keys.apply()
        torch.testing.assert_close(env.command_targets[:, 1], torch.tensor([0.0, 0.2, 0.0]))
        for _ in range(8):
            keys.handle_event(event("q1_yaw_plus"))
        self.assertAlmostEqual(keys.yaw_rate[1].item(), 0.6)
        for _ in range(8):
            keys.handle_event(event("q1_yaw_minus"))
        self.assertAlmostEqual(keys.yaw_rate[1].item(), -0.6)
        torch.testing.assert_close(keys.speed, before)
        keys.apply()
        env.episode_length_buf[:] = 1
        env._post_physics_step_callback()
        yaw_step = env.cfg.commands.yaw_acceleration_limit*env.dt
        self.assertAlmostEqual(env.commands[1, 1].item(), -yaw_step)
        self.assertAlmostEqual(env.compute_proprioception_observations()[1, 7].item(),
                               -yaw_step*env.commands_scale[1].item())
        self.assertTrue((env.command_targets[:, 2] == env.cfg.commands.ranges.height[0]).all())
        keys.handle_event(event("q1_stop"))
        keys.apply()
        self.assertTrue((env.command_targets[:, 1] == 0).all())
        self.assertEqual(keys.speed[1].item(), 0)
        self.assertNotIn("KEY_J", keys.bindings)
        self.assertNotIn("KEY_L", keys.bindings)

    def test_pause_camera_and_handler_cleanup(self):
        env = viewer_env()
        events = []
        env._extra_viewer_action_handler = events.append
        original_render = env.render
        keys = Q1Keyboard(env)
        keys.enable(GYMAPI)
        keys.handle_event(event("q1_select_2"))
        keys.handle_event(event("q1_vx_plus"))
        keys.handle_event(event("q1_pause"))
        self.assertTrue(keys.paused)
        self.assertAlmostEqual(keys.speed[2].item(), 0.2)
        keys.render()
        target = env.root_states[2, :3].tolist()
        self.assertEqual(env.cameras[-1], ([target[0], target[1]+2, target[2]+1], target))
        keys.handle_event(event("q1_free_cam"))
        camera_count = len(env.cameras)
        keys.render()
        self.assertEqual(len(env.cameras), camera_count)
        keys.handle_event(event("q1_pause"))
        self.assertFalse(keys.paused)
        self.assertEqual(len(events), 5)
        keys.close()
        self.assertIs(env.render, original_render)
        self.assertEqual(env._extra_viewer_action_handler, events.append)

    def test_reset_selected_refreshes_history_and_preserves_other_robot(self):
        env = viewer_env()
        keys = Q1Keyboard(env)
        keys.apply()  # activates external command ownership before reset
        env.root_states[0, 0] += 0.1
        other_root = env.root_states[0].clone()
        env.dof_pos[1, env.leg_dof_indices[0]] += 0.4
        env.obs_history[1] = 123.0
        keys.handle_event(event("q1_select_1"))
        keys.handle_event(event("q1_vx_plus"))
        keys.handle_event(event("q1_yaw_plus"))
        with torch.inference_mode():
            env.step(torch.zeros(3, 6))
            keys.handle_event(event("q1_reset"))
            keys.apply()
        torch.testing.assert_close(env.root_states[0], other_root)
        self.assertEqual(keys.speed[1].item(), 0)
        self.assertEqual(keys.yaw_rate[1].item(), 0)
        torch.testing.assert_close(env.obs_buf[1], env.compute_proprioception_observations()[1])
        torch.testing.assert_close(env.obs_history[1], env.obs_buf[1].repeat(env.obs_history_length))
        self.assertEqual(env.episode_length_buf[1].item(), 0)
        self.assertEqual(env.command_targets[1, 0].item(), 0)

    def test_base_renderer_dispatches_events_once_and_keeps_sync_toggle(self):
        base_ns = dict(NS, sys=sys)
        load_code("wheel_legged_gym/envs/base/base_task.py", base_ns)
        base = base_ns["BaseTask"].__new__(base_ns["BaseTask"])
        received = []
        base.viewer, base.sim, base.device = object(), object(), "cpu"
        base.enable_viewer_sync = True
        base._extra_viewer_action_handler = received.append
        events = [event("q1_vx_plus"), event("q1_vx_plus", 0), event("toggle_viewer_sync")]
        base.gym = SimpleNamespace(query_viewer_has_closed=lambda viewer: False,
                                   query_viewer_action_events=lambda viewer: events,
                                   poll_viewer_events=lambda viewer: None)
        base.render()
        self.assertEqual(received, events)
        self.assertFalse(base.enable_viewer_sync)

    def test_keyboard_playback_pauses_before_physics_and_flushes_on_exit(self):
        env = viewer_env(2)
        train = NS["Q1CfgPPO"]()
        frames, slept = [], []
        batches = [[event("q1_pause"), event("q1_vx_plus"), event("q1_yaw_plus")], [event("q1_pause")], None]

        def render():
            frames.append(1)
            batch = batches.pop(0)
            if batch is None:
                raise SystemExit()
            for item in batch:
                env._extra_viewer_action_handler(item)

        env.render = render
        runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=object()),
                                 get_inference_policy=lambda device: lambda obs, history: (torch.zeros(2, 6), None))
        registry = SimpleNamespace(get_cfgs=lambda task: (env.cfg, train),
                                   make_env=lambda *a, **kw: (env, env.cfg),
                                   make_alg_runner=lambda *a, **kw: (runner, train))
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            play_ns = dict(NS, csv=csv, datetime=datetime, json=json, os=os,
                           time=SimpleNamespace(sleep=slept.append), gymapi=GYMAPI,
                           Q1Keyboard=Q1Keyboard, task_registry=registry,
                           export_policy_as_jit=lambda *a: None, WHEEL_LEGGED_GYM_ROOT_DIR=folder)
            load_code("wheel_legged_gym/envs/q1/play_q1.py", play_ns)
            play_ns["play_q1"](SimpleNamespace(num_envs=2, q1_stage="low_speed", load_run="test", checkpoint=1,
                                               headless=False, play_mode="keyboard", play_duration=0.0))
            output = Path(folder)/"logs/q1/exported/evaluations"
            summary = json.loads(next(output.glob("*.json")).read_text())
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["play_mode"], "keyboard")
            self.assertEqual(len(frames), 3)
            self.assertEqual(slept, [0.02])
            self.assertAlmostEqual(env.last_diagnostics[0, 0].item(), 0.2)
            self.assertAlmostEqual(env.command_targets[0, 1].item(), 0.2)
            with next(output.glob("*.csv")).open() as stream:
                self.assertEqual(len(list(csv.reader(stream))), 2)
            self.assertIs(env.render, render)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
