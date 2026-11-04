"""Export the complete history-conditioned policy without simulator dependencies."""

from copy import deepcopy
import inspect
from pathlib import Path
import tempfile

import torch


class SequenceInferencePolicy(torch.nn.Module):
    """Match ActorCriticSequence.act_inference's deterministic action output."""

    def __init__(self, actor_critic):
        super().__init__()
        if not getattr(actor_critic, "is_sequence", False):
            raise ValueError("ONNX export requires an ActorCriticSequence policy")
        # Exporting must not move the live policy off its simulation device or
        # change its training mode. The critic and action noise are not needed.
        self.encoder = deepcopy(actor_critic.encoder).cpu().eval()
        self.actor = deepcopy(actor_critic.actor).cpu().eval()
        self.velocity_estimation_enabled = getattr(actor_critic, "velocity_estimation_enabled", True)
        self.latent_dim = actor_critic.latent_dim

    def forward(self, obs, obs_history):
        if self.velocity_estimation_enabled:
            latent = self.encoder(obs_history)
        else:
            # Keep the public two-input export interface without running the
            # estimator, including ONNX exporters that prune unused inputs.
            latent = obs_history.sum(dim=-1, keepdim=True).expand(-1, self.latent_dim)*0.0
        return self.actor(torch.cat((obs, latent), dim=-1))


def export_policy_as_onnx(actor_critic, path, num_observations, num_history):
    """Write a checked ONNX policy: (obs, flattened history) -> raw actions.

    Batch size is dynamic. Observation scaling, action clipping, CTBC guidance
    and the joint controller remain outside the graph, just as in act_inference.
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires onnx; install it with: python -m pip install onnx") from exc

    model = SequenceInferencePolicy(actor_critic).eval()
    dtype = next(model.parameters()).dtype
    inputs = (torch.zeros(1, num_observations, dtype=dtype),
              torch.zeros(1, num_history, dtype=dtype))
    options = dict(
        export_params=True, opset_version=13,
        input_names=["obs", "obs_history"], output_names=["actions"],
        dynamic_axes={name: {0: "batch"} for name in ("obs", "obs_history", "actions")},
    )
    # Isaac Gym installations also use older torch versions without dynamo.
    # Select the legacy exporter explicitly when newer torch defaults to dynamo.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with torch.no_grad():
            torch.onnx.export(model, inputs, str(temporary), **options)
        onnx.checker.check_model(str(temporary))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
