"""Checkpoint compatibility and explicit CTBC-to-T0 migration without Isaac Gym."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = load_module("q1_t0_run_config", "wheel_legged_gym/envs/q1/t0_run_config.py")
CTBC = load_module("original_ctbc_run_config", "wheel_legged_gym/envs/q1_ctbc/run_config.py")


def snapshot(task="q1", stage="t0"):
    return {"task": task, "stage": stage, "env": {
        "env": {"num_observations": 31, "num_privileged_obs": 151,
                "obs_history_length": 5, "num_actions": 6, "num_envs": 1024},
        "ctbc": {"stage": stage, "feedforward_start_iteration": 30000,
                 "feedforward_hold_iterations": 4321, "feedforward_initial": .7},
        "terrain": {"generation_mode": "t0", "num_rows": 13, "terrain_length": 8.0},
        "control": {"leg_action_scales": [.55, .95, .55, .95]},
        "commands": {"ranges": {"lin_vel_x": [-.7, .7]}},
    }, "train": {
        "seed": 17, "policy": {"num_encoder_obs": 155, "actor_hidden_dims": [128, 128]},
        "runner": {"experiment_name": task, "run_name": "saved_tuning", "resume": False,
                   "max_iterations": 8000, "logger": "tensorboard"},
        "algorithm": {"learning_rate": .000123}}}


def ctbc_snapshot(stage="stairs20"):
    data = snapshot("q1_ctbc", stage)
    data["env"]["terrain"] = {"num_rows": 6, "num_steps": 3,
                               "riser_height_range": [.1, .2]}
    return data


def configs():
    return (SimpleNamespace(
        env=SimpleNamespace(num_envs=4096),
        ctbc=SimpleNamespace(stage="t0", feedforward_start_iteration=0),
        terrain=SimpleNamespace(generation_mode="t0", num_rows=20),
        control=SimpleNamespace(leg_action_scales=[.5, .9, .5, .9]),
        commands=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=[-.8, .8]))),
        SimpleNamespace(seed=1, policy=SimpleNamespace(num_encoder_obs=155),
                        runner=SimpleNamespace(experiment_name="q1", run_name="t0", resume=False,
                                               max_iterations=30000, logger="swanlab"),
                        algorithm=SimpleNamespace(learning_rate=.0005)))


def arguments(**kwargs):
    values = dict(resume=True, q1_stage=None, ctbc_stage=None, run_name=None,
                  experiment_name=None, load_run=None, checkpoint=None,
                  num_envs=None, seed=None, max_iterations=None, logger=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def configure_t0(cfg):
    cfg.ctbc.stage = "t0"
    cfg.ctbc.feedforward_start_iteration = 0
    cfg.terrain.generation_mode = "t0"
    cfg.terrain.num_rows = 20
    cfg.commands.ranges.lin_vel_x = [-.8, .8]
    return cfg


class Q1T0RunConfigTests(unittest.TestCase):
    def setUp(self):
        validate = patch.object(CONFIG, "_validate_ctbc_snapshot", side_effect=CTBC.validate_ctbc_snapshot)
        validate.start()
        self.addCleanup(validate.stop)

    def fixture(self, data=None, name="model_12345.pt"):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP"))
        self.addCleanup(folder.cleanup)
        run = Path(folder.name)
        checkpoint = run / name
        checkpoint.write_bytes(b"checkpoint fixture")
        if data is not None:
            (run / "resolved_config.json").write_text(json.dumps(data), encoding="utf-8")
        return run, checkpoint

    def test_fresh_training_applies_fixed_t0_without_reading_checkpoint(self):
        env, train = configs()
        args, resolver = arguments(resume=False), Mock()
        with patch.object(CONFIG, "_configure_q1_t0", side_effect=configure_t0) as configure:
            result = CONFIG.prepare_q1_t0_training(env, train, args, "unused", resolver)
        configure.assert_called_once_with(env)
        resolver.assert_not_called()
        self.assertEqual(result, (env, train))
        self.assertEqual(env.ctbc.stage, "t0")
        self.assertEqual(env.ctbc.feedforward_start_iteration, 0)
        self.assertEqual(train.runner.ctbc_guidance_start_iteration, 0)
        self.assertEqual((train.runner.experiment_name, train.runner.run_name), ("q1", "t0"))
        self.assertIsNone(args.q1_stage)
        self.assertIsNone(args.ctbc_stage)

    def test_new_q1_resume_restores_tuning_and_guidance_clock_without_reconfiguration(self):
        run, checkpoint = self.fixture(snapshot())
        env, train = configs()
        args, resolver = arguments(), Mock(return_value=str(checkpoint))
        with patch.object(CONFIG, "_configure_q1_t0") as configure, \
                patch.object(CONFIG, "_load_checkpoint_iteration") as reader:
            CONFIG.prepare_q1_t0_training(env, train, args, run, resolver)
        configure.assert_not_called()
        reader.assert_not_called()
        self.assertEqual((env.terrain.num_rows, env.env.num_envs), (13, 1024))
        self.assertEqual(env.control.leg_action_scales, [.55, .95, .55, .95])
        self.assertEqual(env.ctbc.feedforward_start_iteration, 30000)
        self.assertEqual(train.runner.ctbc_guidance_start_iteration, 30000)
        self.assertEqual(train.algorithm.learning_rate, .000123)
        self.assertEqual(train.policy.actor_hidden_dims, [128, 128])
        self.assertEqual(train.runner.run_name, "saved_tuning")
        self.assertEqual(train.runner.experiment_name, "q1")
        self.assertTrue(train.runner.resume)
        self.assertEqual(args.load_run, str(run.resolve()))
        self.assertEqual(args.checkpoint, 12345)
        resolver.assert_called_once_with(str(run / "logs" / "q1"), load_run=-1, checkpoint=-1)

    def test_snapshot_without_clock_uses_global_origin_instead_of_current_config_value(self):
        data = snapshot()
        del data["env"]["ctbc"]["feedforward_start_iteration"]
        run, checkpoint = self.fixture(data)
        env, train = configs()
        env.ctbc.feedforward_start_iteration = 999
        CONFIG.prepare_q1_t0_training(env, train, arguments(), run, Mock(return_value=str(checkpoint)))
        self.assertEqual(env.ctbc.feedforward_start_iteration, 0)
        self.assertEqual(train.runner.ctbc_guidance_start_iteration, 0)

    def test_explicit_absolute_ctbc_migration_preserves_model_contract_but_replaces_terrain(self):
        for old_stage in ("single", "stairs", "robust", "stairs20"):
            with self.subTest(stage=old_stage):
                run, checkpoint = self.fixture(ctbc_snapshot(old_stage), "model_40000.pt")
                env, train = configs()
                args = arguments(load_run=str(run.resolve()))
                resolver = Mock(return_value=str(checkpoint))

                def configure_after_restore(cfg):
                    self.assertEqual(cfg.ctbc.stage, old_stage)
                    self.assertEqual(cfg.terrain.num_rows, 6)
                    return configure_t0(cfg)

                with patch.object(CONFIG, "_configure_q1_t0", side_effect=configure_after_restore) as configure, \
                        patch.object(CONFIG, "_load_checkpoint_iteration", return_value=39999) as reader:
                    CONFIG.prepare_q1_t0_training(env, train, args, "unused", resolver)
                configure.assert_called_once_with(env)
                reader.assert_called_once_with(checkpoint.resolve())
                self.assertEqual((env.ctbc.stage, env.terrain.generation_mode, env.terrain.num_rows),
                                 ("t0", "t0", 20))
                self.assertEqual(env.control.leg_action_scales, [.55, .95, .55, .95])
                self.assertEqual(env.ctbc.feedforward_start_iteration, 39999)
                self.assertEqual(train.runner.ctbc_guidance_start_iteration, 39999)
                self.assertEqual(train.runner.ctbc_guidance_entry_checkpoint_iteration, 39999)
                self.assertEqual(train.runner.q1_t0_source_task, "q1_ctbc")
                self.assertEqual(train.runner.q1_t0_source_checkpoint, str(checkpoint.resolve()))
                self.assertEqual((train.runner.experiment_name, train.runner.run_name), ("q1", "t0"))
                self.assertEqual(train.algorithm.learning_rate, .000123)
                self.assertEqual(train.policy.actor_hidden_dims, [128, 128])
                self.assertTrue(train.runner.resume)
                self.assertEqual(args.checkpoint, 40000)
                resolver.assert_called_once_with(str(run.parent), load_run=str(run.resolve()), checkpoint=-1)

    def test_ctbc_migration_requires_explicit_absolute_source_before_mutation(self):
        run, checkpoint = self.fixture(ctbc_snapshot())
        env, train = configs()
        with patch.object(CONFIG, "_load_checkpoint_iteration") as reader, \
                self.assertRaisesRegex(ValueError, "full absolute --load_run"):
            CONFIG.prepare_q1_t0_training(env, train, arguments(load_run="run_name"), run,
                                          Mock(return_value=str(checkpoint)))
        reader.assert_not_called()
        self.assertEqual(env.env.num_envs, 4096)
        self.assertEqual(train.seed, 1)

    def test_experiment_override_keeps_output_meaning(self):
        run, checkpoint = self.fixture(snapshot())
        env, train = configs()
        args = arguments(experiment_name="q1_custom", load_run="saved", run_name="custom_run",
                         num_envs=64, seed=99, max_iterations=321, logger="swanlab")
        resolver = Mock(return_value=str(checkpoint))
        CONFIG.prepare_q1_t0_training(env, train, args, run, resolver)
        self.assertEqual((train.runner.experiment_name, train.runner.run_name), ("q1_custom", "custom_run"))
        self.assertEqual((args.num_envs, args.seed, args.max_iterations, args.logger), (64, 99, 321, "swanlab"))
        self.assertEqual((env.env.num_envs, train.seed, train.runner.max_iterations, train.runner.logger),
                         (1024, 17, 8000, "tensorboard"))
        resolver.assert_called_once_with(str(run / "logs" / "q1_custom"), load_run="saved", checkpoint=-1)

    def test_ctbc_experiment_override_is_rejected_instead_of_redirecting_output(self):
        env, train = configs()
        resolver = Mock()
        with self.assertRaisesRegex(ValueError, "remove --experiment_name=q1_ctbc"):
            CONFIG.prepare_q1_t0_training(env, train, arguments(experiment_name="q1_ctbc"), "unused", resolver)
        resolver.assert_not_called()

    def test_old_stage_arguments_are_rejected_for_new_q1(self):
        for options, message in (({"q1_stage": "balance"}, "--q1_stage"),
                                 ({"ctbc_stage": "stairs20"}, "--ctbc_stage")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                CONFIG.prepare_q1_t0_training(*configs(), arguments(**options), "unused", Mock())

    def test_legacy_q1_checkpoint_is_rejected_despite_matching_dimensions(self):
        data = snapshot()
        del data["env"]["ctbc"]
        run, checkpoint = self.fixture(data)
        env, train = configs()
        with self.assertRaisesRegex(ValueError, "old q1 checkpoints are not resumable"):
            CONFIG.prepare_q1_t0_training(env, train, arguments(), run, Mock(return_value=str(checkpoint)))
        self.assertEqual(env.env.num_envs, 4096)

    def test_saved_t0_markers_and_model_contract_are_all_required(self):
        cases = []
        for path, value in ((("task",), "q1_flat"), (("stage",), "stairs20"),
                            (("env", "ctbc", "stage"), "stairs"),
                            (("env", "terrain", "generation_mode"), "ctbc"),
                            (("env", "env", "num_observations"), 27),
                            (("env", "env", "num_privileged_obs"), 143),
                            (("env", "env", "obs_history_length"), 4),
                            (("env", "env", "num_actions"), 16),
                            (("train", "policy", "num_encoder_obs"), 135),
                            (("env", "control", "leg_action_scales"), [.5, 0, .5, .9]),
                            (("env", "ctbc", "feedforward_start_iteration"), True)):
            data = snapshot()
            target = data
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            cases.append(data)
        for data in cases + [None, [], {}]:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "incompatible"):
                CONFIG.validate_q1_t0_snapshot(data)
        original = snapshot()
        CONFIG.validate_q1_t0_snapshot(original)
        self.assertEqual(original, snapshot())

    def test_migration_also_rejects_invalid_ctbc_contract_before_mutation(self):
        data = ctbc_snapshot()
        data["env"]["control"]["leg_action_scales"] = [.5, .9, .5]
        run, checkpoint = self.fixture(data)
        env, train = configs()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            CONFIG.prepare_q1_t0_training(env, train, arguments(load_run=str(run.resolve())), run,
                                          Mock(return_value=str(checkpoint)))
        self.assertEqual(env.env.num_envs, 4096)
        self.assertEqual(train.seed, 1)

    def test_missing_snapshot_and_nonstandard_model_names_fail(self):
        run, checkpoint = self.fixture()
        with self.assertRaisesRegex(FileNotFoundError, "saved configuration"):
            CONFIG.prepare_q1_t0_training(*configs(), arguments(), run, Mock(return_value=str(checkpoint)))
        for name in ("policy_latest.pt", "model_0012345.pt"):
            with self.subTest(name=name):
                run, checkpoint = self.fixture(snapshot(), name)
                with self.assertRaisesRegex(ValueError, "model_N.pt"):
                    CONFIG.prepare_q1_t0_training(*configs(), arguments(), run,
                                                  Mock(return_value=str(checkpoint)))

    def test_missing_checkpoint_and_invalid_json_fail_before_mutation(self):
        run, checkpoint = self.fixture(snapshot())
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint does not exist"):
            CONFIG.prepare_q1_t0_training(*configs(), arguments(), run,
                                          Mock(return_value=str(run / "model_99.pt")))
        (run / "resolved_config.json").write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot read q1 configuration"):
            CONFIG.prepare_q1_t0_training(*configs(), arguments(), run, Mock(return_value=str(checkpoint)))

    def test_foreign_task_is_rejected(self):
        data = snapshot("other")
        run, checkpoint = self.fixture(data)
        with self.assertRaisesRegex(ValueError, "from task 'other'"):
            CONFIG.prepare_q1_t0_training(*configs(), arguments(), run, Mock(return_value=str(checkpoint)))

    def test_iteration_reader_uses_safe_cpu_loading_and_real_saved_iteration(self):
        reader = Mock(return_value={"iter": 39999})
        with patch.dict(sys.modules, {"torch": SimpleNamespace(load=reader)}):
            self.assertEqual(CONFIG._load_checkpoint_iteration("model_40000.pt"), 39999)
        reader.assert_called_once_with("model_40000.pt", map_location="cpu", weights_only=True)
        for invalid in (None, True, -1, 1.5, "40000"):
            with self.subTest(iteration=invalid), \
                    patch.dict(sys.modules, {"torch": SimpleNamespace(load=Mock(return_value={"iter": invalid}))}), \
                    self.assertRaisesRegex(ValueError, "checkpoint 'iter'"):
                CONFIG._load_checkpoint_iteration("model_40000.pt")

    def test_invalid_migration_iteration_fails_before_configuration_is_restored(self):
        run, checkpoint = self.fixture(ctbc_snapshot())
        env, train = configs()
        with patch.object(CONFIG, "_load_checkpoint_iteration", side_effect=ValueError("invalid iter")), \
                self.assertRaisesRegex(ValueError, "invalid iter"):
            CONFIG.prepare_q1_t0_training(env, train, arguments(load_run=str(run.resolve())), run,
                                          Mock(return_value=str(checkpoint)))
        self.assertEqual(env.env.num_envs, 4096)
        self.assertEqual(train.seed, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
