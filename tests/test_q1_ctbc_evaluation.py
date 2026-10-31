"""Evaluation accounting tests, independent of Isaac Gym and contact simulation."""
import importlib.util
import ast
import builtins
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "wheel_legged_gym/scripts/evaluate_q1_ctbc.py"
SPEC = importlib.util.spec_from_file_location("evaluate_q1_ctbc", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
RUN_SPEC = importlib.util.spec_from_file_location("ctbc_run_config_for_eval", SCRIPT.parents[1] / "envs/q1_ctbc/run_config.py")
RUN_CONFIG = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN_CONFIG)


class CTBCEvaluationTests(unittest.TestCase):
    def test_play_exports_ctbc_by_default(self):
        tree = ast.parse(SCRIPT.with_name("play.py").read_text(encoding="utf-8"))
        play = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "play")
        namespace = {}
        exec(compile(ast.Module(body=[play], type_ignores=[]), "play.py", "exec"), namespace)
        evaluate = Mock(return_value={"complete": True})
        args = SimpleNamespace(task="q1_ctbc")
        with patch.dict(sys.modules, {
                "wheel_legged_gym.scripts.evaluate_q1_ctbc": SimpleNamespace(evaluate=evaluate)}):
            self.assertEqual(namespace["play"](args), {"complete": True})
        evaluate.assert_called_once_with(args, export_policy=True, plot_torques=True)

    def test_cli_initializes_registry_before_importing_utils(self):
        # A preloaded/mock registry hides the standalone CLI cycle. Reproduce
        # the cold-start dependency: utils cannot finish until envs registers.
        imports = []
        arguments = SimpleNamespace(task="q1_ctbc")
        original_import = builtins.__import__

        def runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "isaacgym":
                imports.append(name)
                return SimpleNamespace()
            if name == "wheel_legged_gym.envs":
                self.assertIn("isaacgym", imports)
                imports.append(name)
                return SimpleNamespace()
            if name == "wheel_legged_gym.utils":
                if "wheel_legged_gym.envs" not in imports:
                    raise ImportError("task_registry is partially initialized")
                imports.append(name)
                return SimpleNamespace(get_args=lambda **kwargs: arguments)
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=runtime_import), \
                patch.object(EVAL, "evaluate", return_value={"complete": True}) as evaluate:
            EVAL.main()
        evaluate.assert_called_once_with(arguments)
        self.assertEqual(imports, ["isaacgym", "wheel_legged_gym.envs", "wheel_legged_gym.utils"])

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

    def test_guidance_metadata_reports_actual_high_lift_and_supports_legacy_environments(self):
        env = SimpleNamespace(
            cfg=SimpleNamespace(ctbc=SimpleNamespace(swing_duration=.8, feedforward_start_iteration=30000),
                                control=SimpleNamespace(leg_action_scales=[.5, .9, .5, .9],
                                                        leg_action_limits=None)),
            ctbc_guide_lift_heights=[.16, .08],
            guidance=SimpleNamespace(amplitudes=[[.55, -1.10], [.3, -.6]]),
            ctbc_leg_limits=[.7, 1.35, .7, 1.35],
            ctbc_action_clip=[1.4, 1.5, 1.0, 1.4, 1.5, 1.0])
        metadata = EVAL.guidance_metadata(env)
        self.assertEqual(metadata["guide_lift_height_m"], .16)
        self.assertEqual(metadata["guide_swing_duration_s"], .8)
        self.assertEqual(metadata["guide_hip_amplitude_rad"], .55)
        self.assertEqual(metadata["guide_knee_amplitude_rad"], -1.10)
        self.assertEqual(metadata["feedforward_start_iteration"], 30000)
        self.assertEqual(metadata["leg_action_scales"], [.5, .9, .5, .9])
        self.assertEqual(metadata["leg_action_limits"], [.7, 1.35, .7, 1.35])
        self.assertEqual(metadata["action_clip_per_channel"], [1.4, 1.5, 1.0, 1.4, 1.5, 1.0])
        json.dumps(metadata)  # metadata must be directly serializable
        self.assertEqual(EVAL.guidance_metadata(SimpleNamespace()), {})
        legacy = SimpleNamespace(cfg=SimpleNamespace(control=SimpleNamespace(leg_action_scales=[.35] * 4)))
        self.assertEqual(EVAL.guidance_metadata(legacy), {"leg_action_scales": [.35] * 4})

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
                self.ctbc_guide_lift_heights = [.16] * self.num_envs
                self.guidance = SimpleNamespace(amplitudes=[[.55, -1.10]] * self.num_envs)
                self.ctbc_leg_limits = [.7, 1.35, .7, 1.35]
                self.ctbc_action_clip = [1.4, 1.5, 1.0, 1.4, 1.5, 1.0]
                self.terrain = SimpleNamespace(riser_heights=Grid(.025), tread_widths=Grid(.5),
                                               step_counts=Grid(1), first_riser_x=Grid(1.7), goal_x=Grid(2.0))
                self.gym = SimpleNamespace(destroy_sim=lambda sim: calls.append("destroy"))

            def set_guidance_mode(self, mode):
                calls.append(("guidance", mode))
                self.ctbc_feedforward_weight = float(mode == "on")

            def set_command_target(self, speed, yaw):
                pass

            def step(self, action):
                calls.append("step")
                recorder = getattr(self, "_play_torque_recorder", None)
                if recorder is not None:
                    recorder.record("physics sample")
                return None, None, None, [True, True, True], {"ctbc_episode": {
                    "env_ids": [0, 1, 2], "success": [True, False, True],
                    "physical_failure": [False, False, False], "timeout": [False, True, False]}}, None

        class Runner:
            def __init__(self, env, config, log_dir, device):
                self.env = env
                self.alg = SimpleNamespace(actor_critic=object())
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
                              ctbc=SimpleNamespace(swing_duration=.8, feedforward_start_iteration=30000),
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
            self.assertEqual([row["feedforward_weight"] for row in report["results"]], [0.0, 1.0])
            self.assertEqual([row["completed"] for row in report["results"]], [2, 2])
            for row in report["results"]:
                self.assertEqual(row["guide_lift_height_m"], .16)
                self.assertEqual(row["guide_swing_duration_s"], .8)
                self.assertEqual(row["feedforward_start_iteration"], 30000)
                self.assertEqual(row["leg_action_limits"], [.7, 1.35, .7, 1.35])
            self.assertEqual(report["results"][0]["checkpoint_iteration"], 12345)
            self.assertEqual(json.loads(Path(args.output).read_text())["seed"], 7)
            self.assertEqual(calls, [("runner", None, 128), ("load", False), ("guidance", "off"), "step", "destroy",
                                     ("runner", None, 128), ("load", False), ("guidance", "on"), "step", "destroy"])

            def export_policy(policy, path, num_observations, num_history):
                calls.append(("export", num_observations, num_history))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"export fixture")
                return path

            modules["wheel_legged_gym.rsl_rl.modules.policy_exporter"] = SimpleNamespace(
                export_policy_as_onnx=export_policy)
            calls.clear()
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                report = EVAL.evaluate(args, export_policy=True)
            exported_path = checkpoint.parent / "exported" / "model_12345" / "policy.onnx"
            self.assertEqual(report["policy_onnx"], str(exported_path))
            self.assertTrue(exported_path.exists())
            self.assertEqual(calls.count(("export", 31, 155)), 1)
            self.assertLess(calls.index(("load", False)), calls.index(("export", 31, 155)))
            self.assertLess(calls.index(("export", 31, 155)), calls.index("step"))

            calls.clear()
            args.export_only = True
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                report = EVAL.evaluate(args)
            self.assertTrue(report["complete"])
            self.assertTrue(report["export_only"])
            self.assertEqual(report["results"], [])
            self.assertEqual(calls, [("runner", None, 128), ("load", False), ("export", 31, 155), "destroy"])

            calls.clear()
            args.export_only = False
            args.export_onnx = True
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()), \
                    patch.object(Env, "step", side_effect=KeyboardInterrupt):
                EVAL.evaluate(args)
            interrupted = json.loads(Path(args.output).read_text())
            self.assertFalse(interrupted["complete"])
            self.assertEqual(interrupted["stop_reason"], "keyboard_interrupt")
            self.assertEqual(interrupted["policy_onnx"], str(exported_path))
            self.assertTrue(exported_path.exists())
            self.assertEqual(calls.count("destroy"), 1)

            class Recorder:
                def __init__(self, env):
                    self.env, self.count = env, 0

                def record(self, torques):
                    self.count += 1

                def save_png(self, path, title):
                    self_test.assertIsNone(self.env._play_torque_recorder)
                    calls.append(("save_plot", self.count))
                    if self.count:
                        path.write_bytes(b"plot fixture")
                        return path

            self_test = self
            modules["wheel_legged_gym.utils.torque_plot"] = SimpleNamespace(TorqueRecorder=Recorder)
            original_step = Env.step

            # Closing the viewer or Ctrl+C must save the accumulated trace and
            # stop, rather than launch the next guidance condition's viewer.
            for signal, reason in ((SystemExit(), "viewer_closed"), (KeyboardInterrupt(), "keyboard_interrupt")):
                with self.subTest(stop=reason):
                    calls.clear()
                    args.episodes = 100

                    def stop_after_one(env, action):
                        if getattr(env, "stepped_once", False):
                            raise signal
                        env.stepped_once = True
                        return original_step(env, action)

                    with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()), \
                            patch.object(Env, "step", stop_after_one):
                        report = EVAL.evaluate(args, plot_torques=True)
                    self.assertEqual(report["stop_reason"], reason)
                    self.assertFalse(report["complete"])
                    self.assertEqual(calls.count("destroy"), 1)
                    self.assertLess(calls.index(("save_plot", 1)), calls.index("destroy"))
                    self.assertEqual(len(report["torque_plots"]), 1)
                    self.assertTrue(Path(report["torque_plots"][0]).exists())
                    self.assertEqual(json.loads(Path(args.output).read_text())["torque_plots"], report["torque_plots"])

            calls.clear()
            args.episodes = 2
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                report = EVAL.evaluate(args, plot_torques=True)
            self.assertTrue(report["complete"])
            self.assertEqual(len(set(report["torque_plots"])), 2)
            self.assertEqual(calls.count(("save_plot", 1)), 2)

            calls.clear()
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()), \
                    patch.object(Env, "step", side_effect=SystemExit):
                report = EVAL.evaluate(args, plot_torques=True)
            self.assertNotIn("torque_plots", report)
            self.assertIn(("save_plot", 0), calls)

            calls.clear()
            args.export_only = True
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                report = EVAL.evaluate(args, plot_torques=True)
            self.assertNotIn("torque_plots", report)
            self.assertEqual(calls, [("runner", None, 128), ("load", False), ("export", 31, 155), "destroy"])
            # A Q1 checkpoint can have the same actor dimensions; reject its
            # missing CTBC contract before creating another environment.
            del saved["env"]["ctbc"]
            config_path.write_text(json.dumps(saved), encoding="utf-8")
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "q1 checkpoints"):
                EVAL.evaluate(args)
            self.assertEqual(calls.count("destroy"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
