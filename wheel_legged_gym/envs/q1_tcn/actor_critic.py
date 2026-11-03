"""Causal history encoder, fixed auxiliary slots and asymmetric actor-critic."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .q1_tcn_config import ABLATIONS


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.ELU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class CausalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return F.elu(x + self.conv(F.pad(x, (self.left_padding, 0))))


class ActorCriticTCN(nn.Module):
    def __init__(self, num_obs, num_critic_obs, num_actions, ablation="A1",
                 history_length=16, channels=64, dilations=(1, 2, 4, 8),
                 kernel_size=2, latent_dim=16, actor_hidden_dims=(128, 64, 32),
                 critic_hidden_dims=(256, 128, 64), init_noise_std=0.5):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError("Unknown ablation: {}".format(ablation))
        if not dilations or min(dilations) < 1 or kernel_size < 2:
            raise ValueError("TCN requires positive dilations and kernel_size >= 2")
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        if self.receptive_field < history_length:
            raise ValueError("TCN receptive field must cover the history window")
        self.num_obs = num_obs
        self.history_length = history_length
        self.ablation = ablation
        self.enabled = ABLATIONS[ablation]
        self.input_projection = nn.Conv1d(num_obs, channels, 1)
        self.tcn = nn.Sequential(*[
            CausalBlock(channels, kernel_size, dilation) for dilation in dilations
        ])
        self.latent_head = nn.Linear(channels, latent_dim)
        self.velocity_head = nn.Linear(channels, 3)
        self.contact_head = nn.Linear(channels, 2)
        self.slip_head = nn.Linear(channels, 2)
        # All ablations retain seven input slots and exactly the same parameters.
        self.actor = build_mlp(num_obs + latent_dim + 7, actor_hidden_dims, num_actions)
        self.critic = build_mlp(num_critic_obs, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())

    def encode(self, history):
        history = history.reshape(-1, self.history_length, self.num_obs).transpose(1, 2)
        h = self.tcn(self.input_projection(history))[:, :, -1]
        return {
            "latent": self.latent_head(h),
            "velocity": self.velocity_head(h),
            "contact_logits": self.contact_head(h),
            "slip": self.slip_head(h),
        }

    def distribution(self, obs, history):
        estimates = self.encode(history)
        slots = [estimates["velocity"], estimates["contact_logits"].sigmoid(), estimates["slip"]]
        slots = [value.detach() if enabled else torch.zeros_like(value)
                 for value, enabled in zip(slots, self.enabled)]
        mean = self.actor(torch.cat([obs, estimates["latent"]] + slots, dim=-1))
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std), estimates

    def evaluate(self, critic_obs):
        return self.critic(critic_obs).squeeze(-1)

    def act_inference(self, obs, history):
        distribution, estimates = self.distribution(obs, history)
        return distribution.mean, estimates


def estimation_losses(estimates, targets, enabled):
    """Targets: v[3], contact[2], slip[2], valid-frame[1], slip-valid[2]."""
    valid = targets[:, 7] > 0.5
    zero = estimates["latent"].sum() * 0.0
    losses = {name: zero for name in ("velocity", "contact", "slip")}
    if enabled[0] and valid.any():
        losses["velocity"] = F.smooth_l1_loss(estimates["velocity"][valid], targets[valid, :3])
    if enabled[1] and valid.any():
        losses["contact"] = F.binary_cross_entropy_with_logits(
            estimates["contact_logits"][valid], targets[valid, 3:5])
    if enabled[2]:
        # Delayed contact classification must not supervise airborne slip.
        mask = valid[:, None] & (targets[:, 8:10] > 0.5)
        if mask.any():
            losses["slip"] = F.smooth_l1_loss(estimates["slip"][mask], targets[:, 5:7][mask])
    return losses
