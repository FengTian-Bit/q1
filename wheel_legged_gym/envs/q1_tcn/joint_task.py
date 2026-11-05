"""Environment and policy changes scoped to the public q1_tcn joint task."""
import torch

from .q1_tcn import Q1TCN, target_sequence_commands
from .actor_critic import ActorCriticTCN


class Q1TCNJoint(Q1TCN):
    def _apply_sequence_commands(self, env_ids=None):
        fraction = getattr(self.cfg.commands, "parking_env_fraction", 0.0)
        if not fraction:
            # Historical experiment entry points explicitly supply their old
            # configs and retain their exact command sampler.
            return super()._apply_sequence_commands(env_ids)
        if getattr(self, "external_commands", False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        count = max(1, min(self.num_envs-1, int(self.num_envs*fraction)))
        parking = env_ids[env_ids < count]
        self.commands[parking, :2] = 0.0
        end = count + int((self.num_envs-count)*self.cfg.commands.sequence_fraction)
        ids = env_ids[(env_ids >= count) & (env_ids < end)]
        pace = .8+.1*(ids % 5).float()
        time_s = self.episode_length_buf[ids].float()*self.dt/pace
        cfg = self.cfg.commands
        vx, wz = target_sequence_commands(time_s, ids,
            min(-cfg.ranges.lin_vel_x[0], cfg.ranges.lin_vel_x[1]),
            min(-cfg.ranges.ang_vel_yaw[0], cfg.ranges.ang_vel_yaw[1]),
            cfg.sequence_min_fraction, cfg.target_edge_fraction, cfg.target_stage)
        self.commands[ids, 0], self.commands[ids, 1] = vx, wz


class JointActorCriticTCN(ActorCriticTCN):
    """Same v7 inference graph, with the parking actor trainable from scratch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.parking_policy is None:
            raise ValueError("Joint training needs a parking policy")
        # The shared privileged critic supplies PPO advantages to both actors.
        # The parking critic/disabled auxiliary heads/noise remain unused.
        for name in ("input_projection", "tcn", "latent_head", "actor"):
            for parameter in getattr(self.parking_policy, name).parameters():
                parameter.requires_grad_(True)
