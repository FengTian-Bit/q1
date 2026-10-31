"""Numerical checks for the deployable sequence policy, without Isaac Gym."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest

import torch

MODULES = Path(__file__).resolve().parents[1] / "wheel_legged_gym/rsl_rl/modules"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, MODULES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ActorCriticSequence = load_module("actor_critic_sequence").ActorCriticSequence
EXPORTER = load_module("policy_exporter")
SequenceInferencePolicy = EXPORTER.SequenceInferencePolicy
export_policy_as_onnx = EXPORTER.export_policy_as_onnx

try:
    import onnx
    import onnxruntime
except ImportError:
    onnx = onnxruntime = None


class PolicyExporterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        with redirect_stdout(io.StringIO()):
            self.policy = ActorCriticSequence(
                31, 154, 6, num_encoder_obs=155, latent_dim=3,
                encoder_hidden_dims=[32, 16], actor_hidden_dims=[32, 16],
                critic_hidden_dims=[16],
            )

    def test_complete_policy_matches_inference_and_keeps_live_model_unchanged(self):
        state = {name: value.clone() for name, value in self.policy.state_dict().items()}
        obs, history = torch.randn(4, 31), torch.randn(4, 155)
        exported = SequenceInferencePolicy(self.policy).eval()
        with torch.no_grad():
            expected, _ = self.policy.act_inference(obs, history)
            torch.testing.assert_close(exported(obs, history), expected)
            # The same current observation with different histories must retain
            # the encoder's influence; exporting actor alone loses this input.
            other, _ = self.policy.act_inference(obs, history + 1)
            self.assertFalse(torch.allclose(expected, other))
            torch.testing.assert_close(exported(obs, history + 1), other)
        self.assertTrue(self.policy.training)
        self.assertTrue(self.policy.actor.training)
        self.assertTrue(self.policy.encoder.training)
        for name, value in self.policy.state_dict().items():
            torch.testing.assert_close(value, state[name])
        self.assertIsNot(exported.actor[0].weight, self.policy.actor[0].weight)

    @unittest.skipIf(onnxruntime is None, "onnx and onnxruntime are needed for ONNX execution")
    def test_onnx_matches_pytorch_for_single_and_multiple_environments(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            path = export_policy_as_onnx(self.policy, Path(folder) / "policy.onnx", 31, 155)
            onnx.checker.check_model(str(path))
            session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.assertEqual([item.name for item in session.get_inputs()], ["obs", "obs_history"])
            self.assertEqual([item.name for item in session.get_outputs()], ["actions"])
            for batch in (1, 4):
                obs, history = torch.randn(batch, 31), torch.randn(batch, 155)
                with torch.no_grad():
                    expected, _ = self.policy.act_inference(obs, history)
                actual = session.run(None, {"obs": obs.numpy(), "obs_history": history.numpy()})[0]
                self.assertEqual(actual.shape, (batch, 6))
                torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
