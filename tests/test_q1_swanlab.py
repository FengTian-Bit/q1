"""Validate SwanLab boundaries and actual PPO logging without cloud credentials."""
from collections import deque
from dataclasses import asdict, is_dataclass
import contextlib
import io
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import optim

from test_q1 import improved
from test_q1_flat import NS, load_code


class TensorBoardStub:
    instances = []

    def __init__(self, log_dir, flush_secs):
        self.scalars = []
        self.closed = 0
        self.instances.append(self)

    def add_scalar(self, tag, value, global_step=None, **kwargs):
        self.scalars.append((tag, float(value), global_step))

    def close(self):
        self.closed += 1


WRITER_NS = dict(NS, os=os, Path=Path, asdict=asdict, is_dataclass=is_dataclass,
                 SummaryWriter=TensorBoardStub)
load_code("wheel_legged_gym/rsl_rl/utils/swanlab_utils.py", WRITER_NS)
SwanLabWriter = WRITER_NS["SwanLabSummaryWriter"]


@contextlib.contextmanager
def use_sdk(module):
    # Restore only this SDK entry; PyTorch lazily registers native modules
    # during PPO updates, and clearing the entire module cache breaks reimport.
    missing = object()
    previous = sys.modules.get("swanlab", missing)
    sys.modules["swanlab"] = module
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("swanlab", None)
        else:
            sys.modules["swanlab"] = previous


def runner_types():
    namespace = dict(NS, os=os, optim=optim, time=time, statistics=statistics, deque=deque,
                     VecEnv=object, SummaryWriter=TensorBoardStub, SwanLabSummaryWriter=SwanLabWriter)
    for path in ("modules/actor_critic.py", "modules/actor_critic_sequence.py", "storage/rollout_storage.py",
                 "algorithms/ppo.py", "runners/on_policy_runner.py"):
        load_code("wheel_legged_gym/rsl_rl/"+path, namespace)
    return namespace["OnPolicyRunner"]


class Q1SwanLabTests(unittest.TestCase):
    def setUp(self):
        TensorBoardStub.instances.clear()
        self.sdk = ModuleType("swanlab")
        self.runs = []

        def init(**kwargs):
            run = SimpleNamespace(id="run_{}".format(len(self.runs)+1), config={}, log=Mock())
            self.runs.append(run)
            return run

        self.sdk.init = Mock(side_effect=init)
        self.sdk.finish = Mock()
        self.addCleanup(patch.stopall)
        sdk_context = use_sdk(self.sdk)
        sdk_context.__enter__()
        self.addCleanup(sdk_context.__exit__, None, None, None)
        patch.dict(os.environ, {"SWANLAB_MODE": "offline", "SWANLAB_WORKSPACE": "robot-team"}).start()

    def make_runner(self, folder, logger="swanlab"):
        cfg = NS["Q1CfgPPO"]()
        train = NS["class_to_dict"](cfg)
        train["runner"].update(num_steps_per_env=2, save_interval=1, logger=logger)
        train["policy"].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8], encoder_hidden_dims=[16, 8])
        train["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
        env = improved(stage="turning")
        env.max_episode_length = 2
        env.get_observations = lambda: (env.obs_buf, env.obs_history)
        env.get_privileged_observations = lambda: env.privileged_obs_buf
        env.reset = lambda: (env.obs_buf, env.privileged_obs_buf)
        with contextlib.redirect_stdout(io.StringIO()):
            return runner_types()(env, train, folder, device="cpu")

    def test_config_init_and_scalar_values_match_local_backup(self):
        cfg = NS["class_to_dict"](NS["Q1CfgPPO"]())["runner"]
        self.assertEqual((cfg["logger"], cfg["swanlab_project"], cfg["swanlab_mode"]), ("swanlab", "Q1", "online"))
        writer = SwanLabWriter("runs/turning", 10, cfg)
        self.addCleanup(writer.close)
        writer.store_config(NS["Q1Cfg"](), {"runner": cfg}, {"start_iteration": 500})
        writer.add_scalar("Episode/q1_yaw_mae", torch.tensor(0.125), 501)
        writer.add_scalar("Loss/encoder", 0.25, 501)
        params = self.sdk.init.call_args.kwargs
        self.assertEqual((params["project"], params["group"], params["mode"]), ("Q1", "turning", "offline"))
        self.assertEqual(params["workspace"], "robot-team")
        self.assertEqual(params["logdir"], os.path.join("runs/turning", "swanlog"))
        self.assertEqual(self.runs[0].config["env_cfg"]["env"]["num_envs"], 4096)
        self.assertEqual(self.runs[0].config["resume"]["start_iteration"], 500)
        self.assertEqual(writer.scalars, [("Episode/q1_yaw_mae", 0.125, 501), ("Loss/encoder", 0.25, 501)])
        self.runs[0].log.assert_any_call({"Episode/q1_yaw_mae": 0.125}, step=501)
        writer.close()
        writer.close()
        self.sdk.finish.assert_called_once()
        self.assertEqual(writer.closed, 1)

    def test_real_ppo_logs_rewards_quality_and_losses_and_links_resume(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            first = self.make_runner(folder)
            self.sdk.init.assert_not_called()  # Construction/playback is logging-free.
            with contextlib.redirect_stdout(io.StringIO()):
                first.learn(2)
            self.assertIsNone(first.writer)
            first_run = self.runs[0]
            metrics = {tag for call in first_run.log.call_args_list for tag in call.args[0]}
            required = {"Episode/q1_yaw_mae", "Episode/q1_slip_speed", "Episode/q1_both_wheel_duty",
                        "Episode/q1_parking_drift", "Loss/value_function", "Loss/encoder", "Loss/surrogate",
                        "Loss/learning_rate", "Policy/mean_noise_std", "Policy/mean_kl", "Perf/total_fps",
                        "Train/mean_reward", "Train/mean_episode_length"}
            self.assertTrue(required.issubset(metrics), required-metrics)
            self.assertEqual({call.kwargs["step"] for call in first_run.log.call_args_list}, {0, 1})
            checkpoint = Path(folder)/"model_2.pt"
            self.assertEqual(torch.load(checkpoint)["swanlab_run_id"], "run_1")
            second = self.make_runner(folder)
            second.load(str(checkpoint))
            with contextlib.redirect_stdout(io.StringIO()):
                second.learn(2)
            self.assertEqual(len(self.runs), 2)
            resumed = self.runs[1]
            self.assertEqual({call.kwargs["step"] for call in resumed.log.call_args_list}, {2, 3})
            self.assertEqual(resumed.config["resume"], {"checkpoint": str(checkpoint.resolve()),
                             "previous_swanlab_run_id": "run_1", "start_iteration": 2})
            self.assertEqual(torch.load(Path(folder)/"model_4.pt")["swanlab_run_id"], "run_2")
            self.assertEqual(self.sdk.finish.call_count, 2)
            self.assertTrue(all(writer.closed == 1 for writer in TensorBoardStub.instances))

    def test_tensorboard_mode_and_old_checkpoints_need_no_swanlab(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            runner = self.make_runner(folder, logger="tensorboard")
            with use_sdk(None), contextlib.redirect_stdout(io.StringIO()):
                runner.learn(1)
            self.sdk.init.assert_not_called()
            checkpoint = Path(folder)/"model_1.pt"
            data = torch.load(checkpoint)
            data.pop("swanlab_run_id")
            torch.save(data, checkpoint)
            runner.load(str(checkpoint))
            self.assertIsNone(runner.loaded_swanlab_run_id)
            self.assertEqual(runner.current_learning_iteration, 1)

    def test_training_exception_closes_both_writers(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            runner = self.make_runner(folder)
            runner.alg.update = Mock(side_effect=RuntimeError("update failed"))
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                runner.learn(1)
            self.sdk.finish.assert_called_once()
            self.assertIsNone(runner.writer)
            self.assertEqual(TensorBoardStub.instances[-1].closed, 1)

    def test_initialization_failure_closes_local_writer_and_missing_sdk_is_explicit(self):
        self.sdk.init.side_effect = RuntimeError("login failed")
        with self.assertRaisesRegex(RuntimeError, "login failed"):
            SwanLabWriter("runs/failed", 10, {"swanlab_project": "Q1"})
        self.assertEqual(TensorBoardStub.instances[-1].closed, 1)
        with use_sdk(None):
            with self.assertRaisesRegex(ModuleNotFoundError, "--logger=tensorboard"):
                SwanLabWriter("runs/missing", 10, {"swanlab_project": "Q1"})

    def test_cli_overrides_resolve_into_training_config(self):
        namespace = dict(NS)
        load_code("wheel_legged_gym/utils/helpers.py", namespace)
        args = SimpleNamespace(seed=None, max_iterations=None, resume=False, experiment_name=None,
                               run_name=None, load_run=None, checkpoint=None, logger="swanlab",
                               swanlab_mode="offline", swanlab_project="my-q1")
        train = NS["Q1CfgPPO"]()
        namespace["update_cfg_from_args"](None, train, args)
        self.assertEqual((train.runner.logger, train.runner.swanlab_mode, train.runner.swanlab_project),
                         ("swanlab", "offline", "my-q1"))
        self.assertEqual((NS["Q1Cfg"]().env.num_envs, train.runner.max_iterations), (4096, 30000))

    def test_episodic_metrics_with_different_keys_are_all_logged(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            runner = self.make_runner(folder, logger="tensorboard")
            writer = TensorBoardStub(folder, 10)
            runner.writer = writer
            values = dict(ep_infos=[{"q1_speed_mae": 0.2}, {"q1_yaw_mae": 0.1}], it=2,
                          collection_time=0.02, learn_time=0.01, mean_value_loss=0.1,
                          mean_surrogate_loss=0.2, mean_extra_loss=0.3, mean_kl=0.01,
                          rewbuffer=deque(), lenbuffer=deque(), start_iter=2, tot_iter=3)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.log(values)
            tags = {tag for tag, value, step in writer.scalars}
            self.assertTrue({"Episode/q1_speed_mae", "Episode/q1_yaw_mae"}.issubset(tags))
            writer.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
