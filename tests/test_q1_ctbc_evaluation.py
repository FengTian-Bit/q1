"""Evaluation accounting tests, independent of Isaac Gym and contact simulation."""
import importlib.util
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "wheel_legged_gym/scripts/evaluate_q1_ctbc.py"
SPEC = importlib.util.spec_from_file_location("evaluate_q1_ctbc", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
RUN_SPEC = importlib.util.spec_from_file_location("ctbc_run_config_for_eval", SCRIPT.parents[1] / "envs/q1_ctbc/run_config.py")
RUN_CONFIG = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN_CONFIG)


class CTBCEvaluationTests(unittest.TestCase):
    def test_height_cli_has_explicit_centimetres_and_rejects_invalid_values(self):
        self.assertEqual(EVAL.parse_heights("2.5, 5,12"), [0.025, 0.05, 0.12])
        for value in ("", "0", "-5", "nan", "inf", "5,,8", "5cm"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EVAL.parse_heights(value)

    def test_final_parallel_batch_is_truncated_in_env_order_to_exact_trial_count(self):
        stats = EVAL.EpisodeAccumulator(2)
        stats.consume({"ctbc_episode": {
            "env_ids": [2, 0, 1], "success": [True, False, True],
            "physical_failure": [False, True, False], "timeout": [False, True, False],
        }}, [True, True, True])
        result = stats.summary()
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["success"], 1)
        self.assertEqual(result["physical_failure"], 1)
        self.assertEqual(result["timeout"], 0)  # physical failure beats simultaneous timeout
        self.assertEqual(result["success_rate"], 0.5)
        self.assertTrue(result["complete"])

    def test_missing_or_misaligned_reset_evidence_cannot_become_success(self):
        with self.assertRaises(RuntimeError):
            EVAL.EpisodeAccumulator(1).consume({}, [True])
        for episode in (
            {"env_ids": [0], "success": [], "physical_failure": [False]},
            {"env_ids": [0, 0], "success": [True, True], "physical_failure": [False, False]},
            {"env_ids": [0], "success": [True], "physical_failure": [True]},
            {"env_ids": [1], "success": [True], "physical_failure": [False]},
            {"env_ids": [], "success": [], "physical_failure": []},
        ):
            with self.subTest(episode=episode), self.assertRaises(ValueError):
                EVAL.EpisodeAccumulator(2).consume({"ctbc_episode": episode}, [True])

    def test_timeout_only_counts_completed_environment(self):
        stats = EVAL.EpisodeAccumulator(1)
        stats.consume({"ctbc_episode": {"env_ids": [1], "success": [False],
                                       "physical_failure": [False]},
                       "time_outs": [False, True]}, [False, True])
        self.assertEqual(stats.summary()["timeout"], 1)
        self.assertEqual(stats.summary()["success_rate"], 0.0)

    def test_rollout_uses_info_snapshot_after_reset_and_stops_at_step_limit(self):
        class ResettingEnv:
            dt = 0.01
            obs_buf, obs_history = object(), object()
            ctbc_success = [False]  # reset already erased the terminal success flag

            def __init__(self, completes):
                self.completes, self.commands, self.steps = completes, [], 0

            def set_command_target(self, speed, yaw):
                self.commands.append((speed, yaw))

            def step(self, actions):
                self.steps += 1
                self.last_action = actions
                infos = {"ctbc_episode": {"env_ids": [0], "success": [True],
                                          "physical_failure": [False], "timeout": [False]}} if self.completes else {}
                return None, None, None, [self.completes], infos, None

        env = ResettingEnv(True)
        result = EVAL.collect_trials(env, lambda obs, history: ("raw_action", None), 3, 0.2, 20)
        self.assertEqual((result["success"], result["completed"], env.steps), (3, 3, 3))
        self.assertEqual(env.last_action, "raw_action")
        self.assertEqual(env.commands, [(0.2, 0.0)] * 3)
        stalled = ResettingEnv(False)
        result = EVAL.collect_trials(stalled, lambda obs, history: "action", 3, 0.2, 5)
        self.assertFalse(result["complete"])
        self.assertEqual(result["policy_steps"], 5)
        self.assertEqual(result["completed"], 0)
        self.assertIsNone(result["success_rate"])

    def test_fixed_selected_geometry_overrides_training_curriculum(self):
        cfg = SimpleNamespace(env=SimpleNamespace(num_envs=4096),
                              terrain=SimpleNamespace(curriculum=True, selected=False),
                              init_state=SimpleNamespace(reset_lin_vel=0.1, reset_ang_vel=0.1))
        EVAL.configure_evaluation(cfg, 0.08, 0.5, 3, 64)
        self.assertFalse(cfg.terrain.curriculum)
        self.assertTrue(cfg.terrain.selected)
        self.assertFalse(cfg.terrain.selected_flat)
        self.assertEqual((cfg.terrain.selected_height, cfg.terrain.selected_tread,
                          cfg.terrain.selected_steps), (0.08, 0.5, 3))
        self.assertEqual((cfg.terrain.num_rows, cfg.terrain.num_cols, cfg.env.num_envs), (1, 1, 64))
        self.assertEqual(cfg.init_state.reset_lin_vel, 0.0)

    def test_checkpoint_config_is_restored_without_optimizers_and_modes_apply_after_load(self):
        calls = []

        class Grid:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, index):
                return self.value

        class Env:
            device, dt, max_episode_length = "cpu", .01, 20
            obs_buf = obs_history = object()
            viewer, sim = None, object()

            def __init__(self, cfg):
                self.cfg, self.num_envs = cfg, cfg.env.num_envs
                self.terrain = SimpleNamespace(riser_heights=Grid(.025), tread_widths=Grid(.5),
                                               step_counts=Grid(1), first_riser_x=Grid(1.7), goal_x=Grid(2.0))
                self.gym = SimpleNamespace(destroy_sim=lambda sim: calls.append("destroy"))

            def set_guidance_mode(self, mode):
                calls.append(("guidance", mode))
                self.ctbc_feedforward_weight = float(mode == "on")

            def set_command_target(self, speed, yaw):
                pass

            def step(self, action):
                return None, None, None, [True, True, True], {"ctbc_episode": {
                    "env_ids": [0, 1, 2], "success": [True, False, True],
                    "physical_failure": [False, False, False], "timeout": [False, True, False]}}, None

        class Runner:
            def __init__(self, env, config, log_dir, device):
                self.env = env
                self.asserted_config = config
                calls.append(("runner", log_dir, config["policy"]["hidden"]))

            def load(self, path, load_optimizer):
                calls.append(("load", load_optimizer))
                self.current_learning_iteration = 12345

            def get_inference_policy(self, device):
                return lambda obs, history: "action"

        cfg = SimpleNamespace(env=SimpleNamespace(num_envs=4096),
                              terrain=SimpleNamespace(curriculum=True, selected=False,
                                                      horizontal_scale=.025, vertical_scale=.005),
                              control=SimpleNamespace(leg_action_scales=[.35] * 4),
                              init_state=SimpleNamespace(reset_lin_vel=.1, reset_ang_vel=.1),
                              noise=SimpleNamespace(add_noise=False), domain_rand=SimpleNamespace(push_robots=False),
                              sim=SimpleNamespace(dt=.005))
        train = SimpleNamespace(runner=SimpleNamespace(experiment_name="q1_ctbc"),
                                seed=1, policy=SimpleNamespace(hidden=256))

        def make_env(name, args, env_cfg):
            self.assertEqual(env_cfg.control.leg_action_scales, [.5, .9, .5, .9])
            self.assertEqual(args.seed, 7)
            self.assertEqual(args.num_envs, 3)
            self.assertFalse(env_cfg.terrain.curriculum)
            return Env(env_cfg), env_cfg

        def config_dict(obj):
            return {key: config_dict(value) for key, value in vars(obj).items()} if isinstance(obj, SimpleNamespace) else obj

        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            checkpoint = Path(folder) / "model_12345.pt"
            saved = {"stage": "single", "env": {
                "env": {"num_observations": 31, "num_privileged_obs": 151,
                        "obs_history_length": 5, "num_actions": 6},
                "ctbc": {"stage": "single"},
                "control": {"leg_action_scales": [.5, .9, .5, .9]}},
                "train": {"policy": {"hidden": 128, "num_encoder_obs": 155}, "runner": {}}}
            config_path = Path(folder) / "resolved_config.json"
            config_path.write_text(json.dumps(saved), encoding="utf-8")
            modules = {
                "isaacgym": SimpleNamespace(),
                "torch": SimpleNamespace(inference_mode=nullcontext),
                "wheel_legged_gym": SimpleNamespace(WHEEL_LEGGED_GYM_ROOT_DIR=folder),
                "wheel_legged_gym.envs": SimpleNamespace(),
                "wheel_legged_gym.utils": SimpleNamespace(task_registry=SimpleNamespace(
                    get_cfgs=lambda name: (cfg, train), make_env=make_env)),
                "wheel_legged_gym.utils.helpers": SimpleNamespace(
                    class_to_dict=config_dict, get_load_path=lambda *args: str(checkpoint)),
                "wheel_legged_gym.envs.q1_ctbc.run_config": RUN_CONFIG,
                "wheel_legged_gym.rsl_rl.runners.on_policy_runner": SimpleNamespace(OnPolicyRunner=Runner),
            }
            args = SimpleNamespace(task="q1_ctbc", heights="2.5", tread=.5, steps=1, episodes=2,
                                   speed=.2, guidance="both", eval_max_steps=0, num_envs=3,
                                   seed=7, experiment_name=None, load_run="run", checkpoint=12345,
                                   rl_device="cpu", output=str(Path(folder) / "report.json"))
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                report = EVAL.evaluate(args)
            self.assertTrue(report["complete"])
            self.assertEqual([row["guidance"] for row in report["results"]], ["off", "on"])
            self.assertEqual([row["completed"] for row in report["results"]], [2, 2])
            self.assertEqual(report["results"][0]["checkpoint_iteration"], 12345)
            self.assertEqual(json.loads(Path(args.output).read_text())["seed"], 7)
            self.assertEqual(calls, [("runner", None, 128), ("load", False), ("guidance", "off"), "destroy",
                                     ("runner", None, 128), ("load", False), ("guidance", "on"), "destroy"])
            # A Q1 checkpoint can have the same actor dimensions; reject its
            # missing CTBC contract before creating another environment.
            del saved["env"]["ctbc"]
            config_path.write_text(json.dumps(saved), encoding="utf-8")
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "q1 checkpoints"):
                EVAL.evaluate(args)
            self.assertEqual(calls.count("destroy"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
