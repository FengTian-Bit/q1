"""Run the public training entry through configuration and artifact recording."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_q1_flat import ROOT, class_to_dict, load_code
from test_q1_ctbc import NAMESPACE


CFG = dict(NAMESPACE, deepcopy=deepcopy)
load_code("wheel_legged_gym/envs/q1/q1_t0_config.py", CFG)
SPEC = importlib.util.spec_from_file_location(
    "q1_t0_entry_run_config", ROOT / "wheel_legged_gym/envs/q1/t0_run_config.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class Q1T0EntryTests(unittest.TestCase):
    def test_training_entry_records_t0_contract_and_all_controller_sources(self):
        cfg, train = CFG["Q1T0Cfg"](), CFG["Q1T0CfgPPO"]()
        learned = []
        run_module = ModuleType("wheel_legged_gym.envs.q1.t0_run_config")
        run_module.prepare_q1_t0_training = RUN.prepare_q1_t0_training
        helpers = ModuleType("wheel_legged_gym.utils.helpers")
        helpers.class_to_dict = class_to_dict
        helpers.get_load_path = lambda *args, **kwargs: self.fail("fresh training loaded a checkpoint")
        args = SimpleNamespace(task="q1", q1_stage=None, ctbc_stage=None, resume=False,
                               experiment_name=None, run_name=None)
        with tempfile.TemporaryDirectory() as directory:
            registry = SimpleNamespace(
                get_cfgs=lambda task: (cfg, train),
                make_env=lambda **kwargs: (SimpleNamespace(), cfg),
                make_alg_runner=lambda **kwargs: (SimpleNamespace(learn=lambda **kw: learned.append(kw)), train),
                save_cfgs=lambda **kwargs: None, log_dir=directory,
            )
            namespace = dict(CFG, os=os, task_registry=registry,
                             WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT))
            load_code("wheel_legged_gym/scripts/train.py", namespace)
            with patch.dict(sys.modules, {run_module.__name__: run_module, helpers.__name__: helpers}), \
                    patch.object(RUN, "_configure_q1_t0", CFG["configure_q1_t0"]):
                namespace["train"](args)
            snapshot = json.loads((Path(directory)/"resolved_config.json").read_text())
            self.assertEqual((snapshot["task"], snapshot["stage"]), ("q1", "t0"))
            self.assertEqual(snapshot["env"]["terrain"]["generation_mode"], "t0")
            self.assertEqual(snapshot["env"]["commands"]["ranges"]["lin_vel_x"], [-1.8, 1.8])
            self.assertEqual(snapshot["env"]["env"]["episode_length_s"], 20.0)
            self.assertEqual(snapshot["train"]["runner"]["experiment_name"], "q1")
            for filename in ("q1.urdf", "source_q1.py", "source_q1_t0.py",
                             "source_q1_t0_config.py", "source_q1_ctbc.py",
                             "source_q1_ctbc_config.py", "source_guidance.py",
                             "source_t0_terrain.py", "source_t0_curriculum.py",
                             "source_t0_commands.py", "source_t0_run_config.py"):
                self.assertTrue((Path(directory)/filename).is_file(), filename)
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0]["num_learning_iterations"], 30000)
        # T0's randomized initial episode clocks remain enabled for the new q1.
        self.assertTrue(learned[0]["init_at_random_ep_len"])

    def test_old_stage_flag_is_rejected_before_creating_a_simulator(self):
        cfg, train = CFG["Q1T0Cfg"](), CFG["Q1T0CfgPPO"]()
        run_module = ModuleType("wheel_legged_gym.envs.q1.t0_run_config")
        run_module.prepare_q1_t0_training = RUN.prepare_q1_t0_training
        helpers = ModuleType("wheel_legged_gym.utils.helpers")
        helpers.get_load_path = lambda *args, **kwargs: None
        registry = SimpleNamespace(get_cfgs=lambda task: (cfg, train),
                                   make_env=lambda **kwargs: self.fail("created simulator"))
        namespace = dict(CFG, task_registry=registry, WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT))
        load_code("wheel_legged_gym/scripts/train.py", namespace)
        args = SimpleNamespace(task="q1", q1_stage="turning", ctbc_stage=None)
        with patch.dict(sys.modules, {run_module.__name__: run_module, helpers.__name__: helpers}):
            with self.assertRaisesRegex(ValueError, "T0 terrain curriculum"):
                namespace["train"](args)


if __name__ == "__main__":
    unittest.main()
