"""Check physical-step torque capture and headless PNG output without Isaac Gym."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_q1_ctbc import ctbc_env


PATH = Path(__file__).resolve().parents[1] / "wheel_legged_gym/utils/torque_plot.py"
SPEC = importlib.util.spec_from_file_location("torque_plot", PATH)
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


class TorquePlotTests(unittest.TestCase):
    def test_every_applied_physics_step_is_recorded_before_reset_in_policy_joint_order(self):
        env = ctbc_env(2, permutation=[5, 2, 4, 0, 3, 1])
        recorder = PLOTS.TorqueRecorder(env)
        env._play_torque_recorder = recorder
        applied = []

        def apply_torque(sim, torques):
            applied.append(torques[0, env.policy_dof_indices].clone())

        env.gym.set_dof_actuation_force_tensor = apply_torque
        # Mimic the terminal reset that would otherwise erase the final sample.
        env.post_physics_step = lambda: env.torques.zero_()
        env.step(torch.full((2, 6), 0.5))
        self.assertEqual(recorder.count, env.cfg.control.decimation)
        self.assertTrue(torch.count_nonzero(torch.stack(applied)).item())
        actual = np.asarray(recorder.samples).reshape(-1, 6)
        np.testing.assert_allclose(actual, torch.stack(applied).numpy())
        self.assertEqual(recorder.joint_names, env.cfg.asset.policy_joint_names)
        self.assertEqual(recorder.dt, env.cfg.sim.dt)
        self.assertEqual(torch.count_nonzero(env.torques).item(), 0)
        self.assertNotEqual(np.count_nonzero(actual[-1]), 0)

    def test_png_is_nonempty_and_empty_sessions_do_not_create_a_file(self):
        from PIL import Image

        env = ctbc_env(1)
        recorder = PLOTS.TorqueRecorder(env)
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            path = Path(folder) / "torques.png"
            self.assertIsNone(recorder.save_png(path))
            self.assertFalse(path.exists())
            # Synthetic input exercises plotting only; it is not a robot run.
            for index in range(100):
                torque = torch.sin(torch.arange(6) + index * 0.1)[None] * 5
                recorder.record(torque)
            saved = recorder.save_png(path)
            self.assertEqual(saved, path.resolve())
            with Image.open(saved) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2240, 1440))
                self.assertGreater(np.asarray(image).std(), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
