"""Exercise the real playback entry with restored configs and a fake simulator."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_q1_t0_run_config import CONFIG, CTBC, snapshot, configs, configure_t0


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q1_t0_play", ROOT / "wheel_legged_gym/envs/q1/play_t0.py"
)
PLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAY)


def plain(value):
    if isinstance(value, SimpleNamespace):
        return {name: plain(item) for name, item in vars(value).items()}
    if isinstance(value, dict):
        return {name: plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def arguments(**overrides):
    values = dict(task="q1", resume=False, q1_stage=None, ctbc_stage=None,
                  experiment_name=None, load_run=None, checkpoint=-1,
                  num_envs=None, headless=True, play_mode="auto",
                  play_duration=0.2, rl_device="cpu", run_name=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class PlaybackFixture:
    def __init__(self, run, data=None, dt=0.1):
        self.calls, self.env, self.controls = [], None, None
        self.run = Path(run)
        self.checkpoint = self.run / "model_12345.pt"
        self.checkpoint.write_bytes(b"not deserialized by playback runner")
        self.data = snapshot() if data is None else data
        (self.run / "resolved_config.json").write_text(json.dumps(self.data), encoding="utf-8")
        cfg, train = configs()
        cfg.env.num_envs = 4096
        cfg.noise = SimpleNamespace(add_noise=True)
        cfg.domain_rand = SimpleNamespace(randomize_friction=True, randomize_base_mass=True,
                                          randomize_action_delay=True, push_robots=True)
        cfg.init_state = SimpleNamespace(reset_lin_vel=0.2, reset_ang_vel=0.3)
        cfg.normalization = SimpleNamespace(clip_observations=100.0)
        cfg.commands.ranges.height = [0.397, 0.397]
        self.cfg, self.train = cfg, train
        self.seen_policy_observations = []
        self.resolver = Mock(return_value=str(self.checkpoint))
        self.registry = SimpleNamespace(get_cfgs=self.get_cfgs, make_env=self.make_env)
        self.runtime = SimpleNamespace(
            torch=torch, gymapi=object(), root=str(self.run), registry=self.registry,
            class_to_dict=plain, get_load_path=self.resolver,
            runner=self.runner, prepare=self.prepare, keyboard=self.keyboard,
        )
        self.dt = dt

    def get_cfgs(self, task):
        self.calls.append("get_cfgs")
        return self.cfg, self.train

    def prepare(self, *args):
        self.calls.append("prepare")
        return CONFIG.prepare_q1_t0_training(*args)

    def make_env(self, task, args, env_cfg):
        self.calls.append("make_env")
        self.assertions = (task, deepcopy(args), deepcopy(env_cfg))
        parent = self

        class FakeEnv:
            device, viewer, sim, num_obs = "cpu", None, "simulation", 3

            def __init__(self):
                self.cfg = env_cfg
                self.num_envs = env_cfg.env.num_envs
                self.dt = parent.dt
                self.commands = torch.zeros(self.num_envs, 3)
                self.command_targets = torch.zeros_like(self.commands)
                self.obs_buf = self.commands.clone()
                self.obs_history = self.obs_buf.repeat(1, 5)
                self.steps = 0
                self.external_commands = False
                self.ctbc_feedforward_weight = 1.0
                self.gym = SimpleNamespace(
                    destroy_sim=lambda sim: parent.calls.append("destroy_sim"),
                    destroy_viewer=lambda viewer: parent.calls.append("destroy_viewer"),
                )

            def set_guidance_mode(self, mode):
                parent.calls.append("guidance_" + mode)
                self.ctbc_feedforward_weight = 0.0 if mode == "off" else 1.0

            def reset(self):
                parent.calls.append("reset")
                if self.ctbc_feedforward_weight != 0:
                    raise AssertionError("Runner/reset received active training guidance")
                self.commands[:] = torch.tensor([0.6, -0.25, 0.397])
                self.obs_buf[:] = self.commands
                self.obs_history[:] = self.obs_buf.repeat(1, 5)
                return self.obs_buf, None

            def compute_proprioception_observations(self):
                return self.commands.clone()

            def set_command_target(self, vx, yaw):
                self.command_targets[:] = torch.tensor([vx, yaw, 0.397])

            def step(self, actions):
                self.steps += 1
                dones = torch.zeros(self.num_envs, dtype=torch.bool)
                infos = {}
                if self.steps == 1:
                    dones[0] = True
                    infos["q1_t0_episode"] = {
                        "env_ids": torch.tensor([0]), "physical_failure": torch.tensor([False]),
                        "timeout": torch.tensor([True]), "distance_m": torch.tensor([3.5]),
                    }
                return self.obs_buf, None, None, dones, infos, self.obs_history

        self.env = FakeEnv()
        return self.env, env_cfg

    def runner(self, env, train_cfg, log_dir, device):
        self.calls.append("runner")
        self.runner_args = deepcopy(train_cfg), log_dir, device
        # Real OnPolicyRunner resets inside its constructor.
        env.reset()
        parent = self

        class FakeRunner:
            current_learning_iteration = 54321

            def load(self, checkpoint, load_optimizer):
                parent.calls.append("load")
                parent.loaded = checkpoint, load_optimizer

            def get_inference_policy(self, device):
                def policy(obs, history):
                    parent.seen_policy_observations.append((obs.clone(), history.clone()))
                    return torch.zeros(env.num_envs, 6), None
                return policy

        return FakeRunner()

    def keyboard(self, env):
        parent = self

        class Controls:
            paused = False

            def enable(self, gymapi):
                parent.calls.append("keyboard_enable")

            def render(self):
                parent.calls.append("keyboard_render")

            def apply(self):
                parent.calls.append("keyboard_apply")
                env.set_command_target(0.7, -0.4)

            def close(self):
                parent.calls.append("keyboard_close")

        self.controls = Controls()
        return self.controls


class T0PlaybackTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP"))
        self.addCleanup(folder.cleanup)
        self.run = Path(folder.name)
        validator = patch.object(CONFIG, "_validate_ctbc_snapshot", side_effect=CTBC.validate_ctbc_snapshot)
        validator.start()
        self.addCleanup(validator.stop)

    def play(self, fixture, args=None, configure=None):
        args = arguments() if args is None else args
        with patch.object(PLAY, "_runtime", return_value=fixture.runtime):
            return PLAY.play_q1(args, playback_configure=configure)

    def test_restores_snapshot_before_env_and_loads_exact_model_without_optimizer_or_logs(self):
        fixture = PlaybackFixture(self.run)
        args = arguments()
        original = deepcopy(plain(fixture.cfg))
        report = self.play(fixture, args)
        self.assertLess(fixture.calls.index("prepare"), fixture.calls.index("make_env"))
        self.assertEqual(fixture.loaded, (str(fixture.checkpoint), False))
        self.assertEqual(fixture.runner_args[1:], (None, "cpu"))
        self.assertEqual(fixture.runner_args[0]["policy"]["actor_hidden_dims"], [128, 128])
        self.assertEqual(fixture.assertions[2].control.leg_action_scales, [.55, .95, .55, .95])
        self.assertEqual(fixture.env.num_envs, 16)
        self.assertFalse(fixture.env.cfg.noise.add_noise)
        self.assertTrue(all(not item for item in vars(fixture.env.cfg.domain_rand).values()))
        self.assertEqual(plain(fixture.cfg), original)
        self.assertFalse(args.resume)
        self.assertIsNone(args.load_run)
        self.assertEqual(report["checkpoint_iteration"], 54321)
        self.assertEqual(report["feedforward_start_iteration"], 30000)
        self.assertEqual(report["feedforward_weight"], 0.0)

    def test_auto_keeps_random_commands_and_reports_distance_episodes(self):
        fixture = PlaybackFixture(self.run)
        report = self.play(fixture)
        self.assertFalse(fixture.env.external_commands)
        first, _ = fixture.seen_policy_observations[0]
        torch.testing.assert_close(first[0], torch.tensor([0.6, -0.25, 0.397]))
        self.assertEqual(report["policy_steps"], 2)
        self.assertEqual(report["completed_episodes"], 1)
        self.assertEqual(report["timeouts"], 1)
        self.assertEqual(report["mean_completed_distance_m"], 3.5)
        self.assertNotIn("success_rate", report)
        saved = list((self.run / "evaluations").glob("q1_t0_*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text())["stage"], "t0")
        self.assertEqual(fixture.calls[-1], "destroy_sim")

    def test_keyboard_applies_direct_commands_before_first_inference_without_double_history_shift(self):
        fixture = PlaybackFixture(self.run)
        report = self.play(fixture, arguments(headless=False, play_mode="keyboard", num_envs=2))
        self.assertTrue(fixture.env.external_commands)
        first, history = fixture.seen_policy_observations[0]
        torch.testing.assert_close(first[0], torch.tensor([0.7, -0.4, 0.397]))
        torch.testing.assert_close(history[0, -3:], first[0])
        # Refreshing a command replaces only the newest frame of history.
        torch.testing.assert_close(history[0, :3], torch.tensor([0.6, -0.25, 0.397]))
        self.assertEqual(fixture.calls.count("keyboard_apply"), 2)
        self.assertIn("keyboard_close", fixture.calls)
        self.assertEqual(report["play_mode"], "keyboard")

    def test_gallery_override_happens_after_restore_and_controls_environment_count(self):
        fixture = PlaybackFixture(self.run)

        def gallery(cfg, args):
            self.assertEqual(cfg.terrain.num_rows, 13)
            cfg.terrain.playback_layout = ["stairs_down", "flat"]
            args.num_envs = 2
            return cfg

        report = self.play(fixture, configure=gallery)
        self.assertEqual(report["num_envs"], 2)
        self.assertEqual(fixture.env.cfg.terrain.playback_layout, ["stairs_down", "flat"])

    def test_headless_forces_auto_and_zero_duration_defaults_to_sixty_seconds(self):
        fixture = PlaybackFixture(self.run, dt=20.0)
        report = self.play(fixture, arguments(headless=True, play_mode="keyboard", play_duration=0))
        self.assertEqual(report["play_mode"], "auto")
        self.assertEqual(report["requested_duration_s"], 60.0)
        self.assertEqual(report["policy_steps"], 3)
        self.assertIsNone(fixture.controls)

    def test_invalid_duration_fails_before_loading_runtime(self):
        for duration in (-1, float("nan"), float("inf")):
            with patch.object(PLAY, "_runtime") as runtime, self.assertRaises(ValueError):
                PLAY.play_q1(arguments(play_duration=duration))
            runtime.assert_not_called()

    def test_legacy_same_shape_q1_checkpoint_rejected_before_simulator(self):
        data = snapshot()
        data["stage"] = "turning"
        data["env"].pop("ctbc")
        fixture = PlaybackFixture(self.run, data)
        with self.assertRaisesRegex(ValueError, "old q1 checkpoints"):
            self.play(fixture)
        self.assertNotIn("make_env", fixture.calls)

    def test_absolute_ctbc_migration_diagnostic_loads_source_without_optimizer(self):
        data = snapshot(task="q1_ctbc", stage="stairs20")
        fixture = PlaybackFixture(self.run, data)
        with patch.object(CONFIG, "_configure_q1_t0", side_effect=configure_t0), \
                patch.object(CONFIG, "_load_checkpoint_iteration", return_value=54321):
            report = self.play(fixture, arguments(load_run=str(self.run)))
        self.assertEqual(report["stage"], "t0")
        self.assertEqual(report["feedforward_start_iteration"], 54321)
        self.assertEqual(fixture.loaded, (str(fixture.checkpoint), False))
        self.assertEqual(fixture.runner_args[0]["runner"]["q1_t0_source_task"], "q1_ctbc")

    def test_episode_counts_ignore_stale_nonterminal_info_and_prefer_physical_failure(self):
        report = dict(completed_episodes=0, physical_failures=0, timeouts=0,
                      other_terminations=0, completed_distance_sum_m=0.0)
        event = {"q1_t0_episode": {"env_ids": [0, 1], "physical_failure": [True, False],
                                   "timeout": [True, True], "distance_m": [1.0, 2.0]}}
        PLAY._collect_episode(report, event, [True, False])
        self.assertEqual(report["completed_episodes"], 1)
        self.assertEqual(report["physical_failures"], 1)
        self.assertEqual(report["timeouts"], 0)
        self.assertEqual(report["completed_distance_sum_m"], 1.0)

    def test_viewer_close_saves_partial_report_and_restores_keyboard(self):
        fixture = PlaybackFixture(self.run)
        original_keyboard = fixture.runtime.keyboard

        def keyboard(env):
            controls = original_keyboard(env)
            controls.render = Mock(side_effect=SystemExit)
            return controls

        fixture.runtime.keyboard = keyboard
        report = self.play(fixture, arguments(headless=False, play_mode="keyboard"))
        self.assertTrue(report["viewer_closed"])
        self.assertEqual(report["policy_steps"], 0)
        self.assertIn("keyboard_close", fixture.calls)
        self.assertEqual(fixture.calls[-1], "destroy_sim")

    def test_misaligned_episode_evidence_fails_explicitly(self):
        event = {"q1_t0_episode": {"env_ids": [0], "physical_failure": [],
                                   "timeout": [True], "distance_m": [2.0]}}
        with self.assertRaisesRegex(ValueError, "misaligned"):
            PLAY._collect_episode({}, event, [True])


if __name__ == "__main__":
    unittest.main()
