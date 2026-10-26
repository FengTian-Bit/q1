"""Pure-CPU checkpoint configuration/provenance tests; no simulator required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "wheel_legged_gym/envs/q1_ctbc/run_config.py"
SPEC = importlib.util.spec_from_file_location("ctbc_run_config", SOURCE)
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def snapshot(stage="stairs"):
    return {"stage": stage, "env": {
        "env": {"num_observations": 31, "num_privileged_obs": 151, "obs_history_length": 5,
                "num_actions": 6, "num_envs": 1024},
        "ctbc": {"stage": stage, "feedforward_initial": .7, "contact_threshold": 15,
                 "feedforward_hold_iterations": 4321, "feedforward_anneal_iterations": 23456},
        "control": {"leg_action_scales": [.55, .95, .55, .95]},
        "terrain": {"num_steps": 3, "riser_height_range": [.045, .115]},
        "domain_rand": {"friction_range": [.65, 1.05]},
        "commands": {"ranges": {"lin_vel_x": [.16, .24]}},
    }, "train": {
        "seed": 17, "policy": {"num_encoder_obs": 155, "actor_hidden_dims": [128, 128]},
        "runner": {"experiment_name": "q1_ctbc", "run_name": "tuned_stairs", "max_iterations": 8000,
                   "logger": "tensorboard", "resume": False},
        "algorithm": {"learning_rate": .000123}}}


def configs():
    return (SimpleNamespace(env=SimpleNamespace(num_envs=4096),
                            ctbc=SimpleNamespace(stage="single", feedforward_hold_iterations=3000),
                            control=SimpleNamespace(leg_action_scales=[.5, .9, .5, .9]),
                            terrain=SimpleNamespace(num_steps=1),
                            domain_rand=SimpleNamespace(friction_range=[.6, 1.2]),
                            commands=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=[.15, .3]))),
            SimpleNamespace(seed=1, policy=SimpleNamespace(num_encoder_obs=155, actor_hidden_dims=[256, 128]),
                            runner=SimpleNamespace(experiment_name="q1_ctbc", run_name="single",
                                                   max_iterations=30000, logger="swanlab"),
                            algorithm=SimpleNamespace(learning_rate=.0005)))


def arguments(**kwargs):
    values = dict(resume=True, ctbc_stage=None, run_name=None, experiment_name=None,
                  load_run=None, checkpoint=None, num_envs=None, max_iterations=None,
                  seed=None, logger=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def configure_stage(cfg, stage):
    cfg.ctbc.stage = stage
    cfg.terrain.num_steps = 1 if stage == "single" else 3
    cfg.terrain.riser_height_range = [.025, .08] if stage == "single" else [.04, .12]
    cfg.commands.ranges.lin_vel_x = [.15, .3]
    return cfg


class CTBCRunConfigTests(unittest.TestCase):
    def fixture(self, data=None, checkpoint_name="model_12345.pt"):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP"))
        self.addCleanup(folder.cleanup)
        run = Path(folder.name)
        checkpoint = run / checkpoint_name
        checkpoint.write_bytes(b"checkpoint fixture")
        if data is not None:
            (run / "resolved_config.json").write_text(json.dumps(data), encoding="utf-8")
        return run, checkpoint

    def test_resume_without_stage_restores_tuning_and_pins_resolved_checkpoint(self):
        run, checkpoint = self.fixture(snapshot())
        env, train = configs()
        args = arguments()
        resolver = Mock(return_value=str(checkpoint))
        with patch.object(CONFIG, "_configure_stage") as stage:
            result = CONFIG.prepare_ctbc_training(env, train, args, run, resolver)
        stage.assert_not_called()
        self.assertIs(result[0], env)
        self.assertIs(result[1], train)
        self.assertEqual(args.ctbc_stage, "stairs")
        self.assertEqual(args.load_run, str(run.resolve()))
        self.assertEqual(args.checkpoint, 12345)
        self.assertEqual(args.run_name, "stairs")
        self.assertEqual(env.ctbc.feedforward_hold_iterations, 4321)
        self.assertEqual(env.ctbc.feedforward_anneal_iterations, 23456)
        self.assertEqual(env.ctbc.feedforward_initial, .7)
        self.assertEqual(env.control.leg_action_scales, [.55, .95, .55, .95])
        self.assertEqual(env.commands.ranges.lin_vel_x, [.16, .24])
        self.assertEqual(env.terrain.riser_height_range, [.045, .115])
        self.assertEqual(train.policy.actor_hidden_dims, [128, 128])
        self.assertEqual(train.algorithm.learning_rate, .000123)
        self.assertTrue(train.runner.resume)
        resolver.assert_called_once_with(str(run / "logs" / "q1_ctbc"), load_run=-1, checkpoint=-1)

    def test_explicit_same_stage_does_not_reset_user_adjustments(self):
        run, checkpoint = self.fixture(snapshot("robust"))
        env, train = configs()
        args = arguments(ctbc_stage="robust", run_name="my_refinement")
        with patch.object(CONFIG, "_configure_stage") as stage:
            CONFIG.prepare_ctbc_training(env, train, args, run, Mock(return_value=str(checkpoint)))
        stage.assert_not_called()
        self.assertEqual(args.run_name, "my_refinement")
        self.assertEqual(train.runner.run_name, "my_refinement")
        self.assertEqual(env.domain_rand.friction_range, [.65, 1.05])

    def test_explicit_different_stage_changes_curriculum_after_restore(self):
        run, checkpoint = self.fixture(snapshot("single"))
        env, train = configs()
        args = arguments(ctbc_stage="stairs")
        with patch.object(CONFIG, "_configure_stage", side_effect=configure_stage) as stage:
            CONFIG.prepare_ctbc_training(env, train, args, run, Mock(return_value=str(checkpoint)))
        stage.assert_called_once_with(env, "stairs")
        self.assertEqual(env.terrain.num_steps, 3)
        self.assertEqual(env.terrain.riser_height_range, [.04, .12])
        # Stage switching does not reset an explicitly tuned annealing horizon.
        self.assertEqual(env.ctbc.feedforward_anneal_iterations, 23456)

    def test_other_cli_overrides_remain_for_standard_registry_handling(self):
        run, checkpoint = self.fixture(snapshot())
        env, train = configs()
        args = arguments(num_envs=64, seed=99, max_iterations=321, logger="swanlab",
                         experiment_name="custom_experiment", load_run="oldrun", checkpoint=12345)
        resolver = Mock(return_value=str(checkpoint))
        CONFIG.prepare_ctbc_training(env, train, args, run, resolver)
        self.assertEqual((args.num_envs, args.seed, args.max_iterations, args.logger), (64, 99, 321, "swanlab"))
        self.assertEqual((env.env.num_envs, train.seed, train.runner.max_iterations, train.runner.logger),
                         (1024, 17, 8000, "tensorboard"))
        resolver.assert_called_once_with(str(run / "logs" / "custom_experiment"), load_run="oldrun", checkpoint=12345)

    def test_fresh_training_defaults_to_single_and_uses_requested_name(self):
        for requested, expected in ((None, "single"), ("stairs", "stairs")):
            with self.subTest(requested=requested):
                env, train = configs()
                args = arguments(resume=False, ctbc_stage=requested)
                resolver = Mock()
                with patch.object(CONFIG, "_configure_stage", side_effect=configure_stage) as stage:
                    CONFIG.prepare_ctbc_training(env, train, args, "unused", resolver)
                stage.assert_called_once_with(env, expected)
                resolver.assert_not_called()
                self.assertEqual(args.ctbc_stage, expected)
                self.assertEqual(train.runner.run_name, expected)

    def test_missing_snapshot_or_foreign_task_fails_before_mutation(self):
        run, checkpoint = self.fixture()
        env, train = configs()
        with self.assertRaisesRegex(FileNotFoundError, "saved configuration"):
            CONFIG.prepare_ctbc_training(env, train, arguments(), run, Mock(return_value=str(checkpoint)))
        data = snapshot()
        del data["env"]["ctbc"]
        (run / "resolved_config.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "q1 checkpoints"):
            CONFIG.prepare_ctbc_training(env, train, arguments(), run, Mock(return_value=str(checkpoint)))
        self.assertEqual(env.env.num_envs, 4096)
        self.assertEqual(train.seed, 1)

    def test_contract_rejects_missing_or_incompatible_dimensions_and_action_scales(self):
        cases = []
        for key, value in (("num_observations", 27), ("num_privileged_obs", 143),
                           ("obs_history_length", 4), ("num_actions", 8)):
            data = snapshot()
            data["env"]["env"][key] = value
            cases.append(data)
        data = snapshot()
        data["train"]["policy"]["num_encoder_obs"] = 135
        cases.append(data)
        for scales in (None, [1, 1, 1], [1, 0, 1, 1], [1, float("nan"), 1, 1]):
            data = snapshot()
            data["env"]["control"]["leg_action_scales"] = scales
            cases.append(data)
        data = snapshot()
        data["stage"] = "robust"
        cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "incompatible"):
                CONFIG.validate_ctbc_snapshot(data)

    def test_nonstandard_checkpoint_name_is_not_silently_reselected(self):
        for name in ("policy_latest.pt", "model_0012345.pt"):
            with self.subTest(name=name):
                run, checkpoint = self.fixture(snapshot(), name)
                with self.assertRaisesRegex(ValueError, "model_N.pt"):
                    CONFIG.prepare_ctbc_training(*configs(), arguments(), run, Mock(return_value=str(checkpoint)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
