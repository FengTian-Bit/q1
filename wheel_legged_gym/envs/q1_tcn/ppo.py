"""PPO with aligned supervised targets and one optimizer for the TCN trunk."""

import torch

from .actor_critic import estimation_losses


def compute_gae(rewards, dones, values, last_values, gamma, lam):
    """Timeout rewards are bootstrapped before this function; dones cut GAE."""
    returns = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(len(rewards))):
        next_value = last_values if step == len(rewards) - 1 else values[step + 1]
        continuation = (~dones[step]).float()
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        advantage = delta + gamma * lam * continuation * advantage
        returns[step] = advantage + values[step]
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    return returns, advantages


class PPO:
    def __init__(self, actor_critic, learning_rate=5e-4, num_learning_epochs=5,
                 num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95,
                 value_loss_coef=1.0, entropy_coef=0.01, max_grad_norm=1.0,
                 desired_kl=0.01, velocity_loss_coef=1.0, contact_loss_coef=1.0,
                 slip_loss_coef=1.0):
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma, self.lam = gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.max_grad_norm, self.desired_kl = max_grad_norm, desired_kl
        self.loss_weights = dict(velocity=velocity_loss_coef, contact=contact_loss_coef,
                                 slip=slip_loss_coef)

    def update(self, batch, last_values):
        model = self.actor_critic
        returns, advantages = compute_gae(batch["rewards"], batch["dones"], batch["values"],
                                          last_values, self.gamma, self.lam)
        data = {key: value.flatten(0, 1) for key, value in batch.items()}
        returns, advantages = returns.flatten(), advantages.flatten()
        count = len(returns)
        if not 1 <= self.num_mini_batches <= count:
            raise ValueError("num_mini_batches must be between 1 and rollout sample count")
        totals, updates = {}, 0
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(count, device=returns.device)
            for ids in indices.tensor_split(self.num_mini_batches):
                distribution, estimates = model.distribution(data["obs"][ids], data["history"][ids])
                log_prob = distribution.log_prob(data["actions"][ids]).sum(-1)
                ratio = (log_prob - data["log_prob"][ids]).exp()
                surrogate = torch.maximum(-advantages[ids] * ratio,
                    -advantages[ids] * ratio.clamp(1.0-self.clip_param, 1.0+self.clip_param)).mean()
                value = model.evaluate(data["critic_obs"][ids])
                clipped = data["values"][ids] + (value-data["values"][ids]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((value-returns[ids]).square(), (clipped-returns[ids]).square()).mean()
                aux = estimation_losses(estimates, data["targets"][ids], model.enabled)
                entropy = distribution.entropy().sum(-1).mean()
                loss = surrogate + self.value_loss_coef*value_loss - self.entropy_coef*entropy
                loss = loss + sum(self.loss_weights[name]*term for name, term in aux.items())
                with torch.no_grad():
                    old_std, old_mean = data["std"][ids], data["mean"][ids]
                    kl = (torch.log(distribution.stddev/old_std) +
                          (old_std.square()+(old_mean-distribution.mean).square()) /
                          (2*distribution.stddev.square()) - 0.5).sum(-1).mean()
                    if self.desired_kl:
                        if kl > 2.0*self.desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate/1.5)
                        elif kl < 0.5*self.desired_kl:
                            self.learning_rate = min(1e-3, self.learning_rate*1.5)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.learning_rate
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Q1 TCN PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                metrics = dict(policy_loss=surrogate, value_loss=value_loss, entropy=entropy, kl=kl,
                               **{name+"_loss": term for name, term in aux.items()})
                for name, term in metrics.items():
                    totals[name] = totals.get(name, 0.0) + term.item()
                updates += 1
        return {name: value/updates for name, value in totals.items()}
