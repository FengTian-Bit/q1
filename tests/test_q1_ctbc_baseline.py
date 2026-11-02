"""Keep a fresh CTBC launch aligned with the user's saved stairs experiment.

The fixture is the unmodified desktop log/resolved_config.json, copied on
2026-10-31. This checks configuration equivalence, not physical performance.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from test_q1_flat import NS
from test_q1_ctbc import NAMESPACE
from test_q1_ctbc_run_config import CONFIG, arguments
from test_q1_ctbc_highcom import HIGH


FIXTURE = Path(__file__).parent / "fixtures/q1_ctbc_stairs_resolved.json"


class CTBCSavedBaselineTests(unittest.TestCase):
    def test_fresh_default_launch_matches_all_saved_training_parameters(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        env, train = NAMESPACE["Q1CTBCCfg"](), NAMESPACE["Q1CTBCCfgPPO"]()
        args = arguments(resume=False)
        resolver = Mock()
        CONFIG.prepare_ctbc_training(
            env, train, args, "unused", resolver,
            stage_configurator=NAMESPACE["configure_ctbc_stage"])
        resolver.assert_not_called()
        # TaskRegistry.get_cfgs copies the training seed to the environment.
        env.seed = train.seed
        actual = json.loads(json.dumps({
            "task": "q1_ctbc", "stage": args.ctbc_stage,
            "env": NS["class_to_dict"](env),
            "train": NS["class_to_dict"](train)}))
        # These two fields are added by simulator/terrain construction. The
        # source seconds, dt, decimation and terrain dimensions remain checked.
        expected["env"]["domain_rand"].pop("push_interval")
        expected["env"]["terrain"].pop("num_sub_terrains")
        # Production serialization includes this empty method artifact.
        for section in ("env", "train"):
            expected[section].pop("init_member_classes")
        # Additive diagnostics stay off in the saved stairs baseline. The
        # historical fixture remains untouched; all control parameters match.
        metadata = {key: actual["env"]["ctbc"].pop(key) for key in (
            "guidance_profile", "diagnostics_enabled", "stall_timeout_s",
            "stall_progress_distance", "stall_support_rise")}
        self.assertEqual(metadata, dict(guidance_profile="legacy", diagnostics_enabled=False,
                                       stall_timeout_s=3.0, stall_progress_distance=.03,
                                       stall_support_rise=.02))
        self.assertEqual(actual, expected)
        self.assertEqual(env.env.num_envs * train.runner.num_steps_per_env, 49152)

    def test_single_stage_remains_an_explicit_training_option(self):
        env, train = NAMESPACE["Q1CTBCCfg"](), NAMESPACE["Q1CTBCCfgPPO"]()
        args = arguments(resume=False, ctbc_stage="single")
        CONFIG.prepare_ctbc_training(
            env, train, args, "unused", Mock(),
            stage_configurator=NAMESPACE["configure_ctbc_stage"])
        self.assertEqual((args.ctbc_stage, train.runner.run_name), ("single", "single"))
        self.assertEqual(env.env.num_envs, 1024)
        self.assertEqual(env.terrain.num_steps, 1)
        self.assertEqual(env.terrain.riser_height_range, [.025, .08])

    def test_highcom_keeps_its_existing_default_stage_and_batch(self):
        env, train = HIGH["Q1CTBCHighCOMCfg"](), HIGH["Q1CTBCHighCOMCfgPPO"]()
        args = arguments(resume=False)
        HIGH["prepare_highcom_training"](env, train, args, "unused", Mock())
        self.assertEqual((args.ctbc_stage, train.runner.run_name), ("single", "single"))
        self.assertEqual(env.env.num_envs, 4096)
        self.assertEqual(env.terrain.num_steps, 1)
        self.assertEqual(NS["Q1Cfg"]().env.num_envs, 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
