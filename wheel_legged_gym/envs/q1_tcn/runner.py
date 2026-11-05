"""Explicit rollout collection, checkpoint metadata and readable CSV logging."""

import csv
import json
import random
import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .actor_critic import ActorCriticTCN
from .ppo import PPO


def config_dict(obj):
    if isinstance(obj, dict):
        return {key: config_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [config_dict(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return {key: config_dict(getattr(obj, key)) for key in dir(obj)
            if not key.startswith("_") and not callable(getattr(obj, key))}


def restore_config(obj, values):
    """Restore the saved configuration without silently accepting schema drift."""
    for name, value in values.items():
        if not hasattr(obj, name):
            raise ValueError("Unknown saved configuration field: {}".format(name))
        if isinstance(value, dict) and isinstance(getattr(obj, name), dict):
            setattr(obj, name, deepcopy(value))
        elif isinstance(value, dict):
            restore_config(getattr(obj, name), value)
        else:
            setattr(obj, name, value)


def load_checkpoint(path, device="cpu"):
    # Older PyTorch versions used with Isaac Gym do not accept weights_only.
    options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(path, **options)


class TCNPpoRunner:
    def __init__(self, env, train_cfg, log_dir, device="cpu"):
        self.env, self.cfg = env, train_cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = ActorCriticTCN(env.num_obs, env.num_privileged_obs, env.num_actions,
            ablation=train_cfg.ablation, **config_dict(train_cfg.policy)).to(self.device)
        self.alg = PPO(self.model, **config_dict(train_cfg.algorithm))
        self.iteration = 0
        self.total_steps = 0
        self.snapshot = {"task": "q1_tcn", "schema_version": 1,
                         "env": config_dict(env.cfg), "train": config_dict(train_cfg)}
        (self.log_dir / "resolved_config.json").write_text(
            json.dumps(self.snapshot, indent=2), encoding="utf-8")

    def collect(self):
        env, model = self.env, self.model
        batch = {}
        failures, timeouts = 0, 0
        reward_metrics = {}
        action_clipping = 0.0
        acceleration_peaks = {}
        with torch.no_grad():
            for _ in range(self.cfg.runner.num_steps_per_env):
                obs, history = env.get_observations()
                obs, history = obs.to(self.device).clone(), history.to(self.device).clone()
                critic_obs = env.get_privileged_observations().to(self.device).clone()
                targets = env.estimator_targets.to(self.device).clone()
                distribution, _ = model.distribution(obs, history)
                actions = distribution.sample()
                values = model.evaluate(critic_obs)
                _, _, rewards, dones, infos, _ = env.step(actions.to(env.device))
                for name, term in env.reward_terms.items():
                    reward_metrics[name] = reward_metrics.get(name, 0.0) + term.mean()
                action_clipping += env.transition_metrics["action_clipping"].mean()
                for name in ['command_acceleration','linear_acceleration']:
                    if name in env.transition_metrics:
                        peak=env.transition_metrics[name].abs().max()
                        acceleration_peaks[name]=torch.maximum(acceleration_peaks[name],peak) if name in acceleration_peaks else peak
                rewards, dones = rewards.to(self.device).clone(), dones.to(self.device).bool()
                environment_rewards = rewards.clone()
                timeout = infos.get("time_outs", torch.zeros_like(dones)).to(self.device).bool()
                # Use the terminal state, never the automatically reset next state.
                # A physical failure simultaneous with timeout is not bootstrapped.
                failed = env.transition_metrics["failure"].to(self.device)
                timeout = timeout & ~failed
                terminal_values = model.evaluate(env.terminal_critic_obs.to(self.device))
                rewards += self.alg.gamma * terminal_values * timeout
                failures += failed.sum()
                timeouts += timeout.sum()
                transition = dict(obs=obs, history=history, critic_obs=critic_obs, targets=targets,
                    actions=actions, values=values, log_prob=distribution.log_prob(actions).sum(-1),
                    mean=distribution.mean, std=distribution.stddev, rewards=rewards, dones=dones,
                    environment_rewards=environment_rewards)
                for name, value in transition.items():
                    batch.setdefault(name, []).append(value.clone())
            last_values = model.evaluate(env.get_privileged_observations().to(self.device))
        steps = self.cfg.runner.num_steps_per_env
        self.rollout_reward_metrics = {"reward_"+name: float(value/steps) for name, value in reward_metrics.items()}
        self.rollout_action_clipping = float(action_clipping/steps)
        self.rollout_acceleration_peaks = {name+'_max':float(value) for name,value in acceleration_peaks.items()}
        return {key: torch.stack(value) for key, value in batch.items()}, last_values, int(failures), int(timeouts)

    def save(self):
        path = self.log_dir / "model_{}.pt".format(self.iteration)
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iteration": self.iteration, "total_steps": self.total_steps,
                    "learning_rate": self.alg.learning_rate, "config": self.snapshot,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)
        return path

    def load(self, path, allow_symmetry_update=False, allow_tracking_update=False,
             allow_yaw_weight_update=False, allow_wheel_diff_update=False):
        checkpoint = load_checkpoint(path, self.device)
        saved = checkpoint["config"]
        # Resume must preserve conditions and capacity, not just matching tensor shapes.
        for section in ("env", "train"):
            previous, current = deepcopy(saved[section]), deepcopy(self.snapshot[section])
            if section == 'env':
                previous['rewards'].setdefault('relative_tracking',{})
                previous['rewards'].setdefault('wheel_diff_tracking',{})
                previous['rewards']['scales'].setdefault('tracking_wheel_diff',0.)
                if allow_tracking_update:
                    if previous['rewards']['reward_version'] not in (6,7) or current['rewards']['reward_version']!=7:
                        raise ValueError('Tracking update only supports reward v6/v7 to v7')
                    previous['rewards']['reward_version']=current['rewards']['reward_version']
                    previous['rewards']['relative_tracking']=current['rewards']['relative_tracking']
                if allow_yaw_weight_update:
                    old_scales=previous['rewards']['scales']
                    new_scales=current['rewards']['scales']
                    linear=new_scales['tracking_lin_vel']
                    if (not np.isfinite(linear) or linear<=0 or old_scales['tracking_lin_vel']!=linear
                            or new_scales['tracking_ang_vel']!=.5*linear):
                        raise ValueError('Yaw weight update must preserve linear weight and set angular weight to half')
                    old_scales['tracking_ang_vel']=new_scales['tracking_ang_vel']
                if allow_wheel_diff_update:
                    settings=current['rewards']['wheel_diff_tracking']
                    weight=current['rewards']['scales']['tracking_wheel_diff']
                    if (set(settings)!={'track_width','sigma'} or weight<=0 or not np.isfinite(weight)
                            or any(not np.isfinite(v) or v<=0 for v in settings.values())):
                        raise ValueError('Wheel guidance update needs positive weight, track_width and sigma')
                    previous['rewards']['wheel_diff_tracking']=deepcopy(settings)
                    previous['rewards']['scales']['tracking_wheel_diff']=weight
                previous['normalization'].setdefault('leg_clip_actions', None)
                previous['commands'].setdefault('sequence_fraction', 0.)
                previous['commands'].setdefault('sequence_min_fraction', .85)
                previous['commands'].setdefault('target_mode',False)
                previous['commands'].setdefault('max_linear_acceleration',0.0)
                previous['commands'].setdefault('target_stage','linear')
                previous['commands'].setdefault('target_edge_fraction',.25)
            if section == 'env' and 'odometry' in previous:
                previous['odometry'].setdefault('observe_wheel_increments',False)
                previous['odometry'].setdefault('increment_scale',.1)
                previous['odometry'].setdefault('parking_observation_delay_s',0.)
                previous['odometry'].setdefault('parking_blend_ramp_s',0.)
            if section == "train":
                previous['policy'].setdefault('command_reference', {})
                previous['policy'].setdefault('wheel_residual', False)
                previous['policy'].setdefault('wheel_residual_limit', .3)
                previous['policy'].setdefault('wheel_residual_yaw_scale',0.)
                previous['policy'].setdefault('steering_feedforward',{})
                previous['policy'].setdefault('motion_wheel_gain',{})
                previous['policy'].setdefault('parking_policy', False)
                previous['policy'].setdefault('parking_blend', False)
                previous['policy'].setdefault('freeze_motion_backbone', True)
                previous['policy'].setdefault('parking_settle', {})
                previous['policy'].setdefault('parking_fractional_flag', False)
                previous['policy'].setdefault('braking_wheel_extension', {})
                previous['algorithm'].setdefault('max_kl',None)
                previous['algorithm'].setdefault('symmetry_loss_coef',0.0)
                previous['algorithm'].setdefault('symmetry_batch_size',4096)
                if allow_symmetry_update:
                    # Explicit target-training continuation may add this one
                    # loss. Observation, control, reward and ranges stay strict.
                    previous['algorithm']['symmetry_loss_coef']=current['algorithm']['symmetry_loss_coef']
                previous["runner"].pop("max_iterations")
                current["runner"].pop("max_iterations")
            if previous != current:
                raise ValueError("Resume configuration mismatch in {}".format(section))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.alg.learning_rate = checkpoint["learning_rate"]
        self.iteration, self.total_steps = checkpoint["iteration"], checkpoint["total_steps"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

    def learn(self, iterations):
        self.model.train()
        csv_path = self.log_dir / "training.csv"
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = None
            for _ in range(iterations):
                batch, last_values, failures, timeouts = self.collect()
                metrics = self.alg.update(batch, last_values)
                self.iteration += 1
                self.total_steps += self.env.num_envs*self.cfg.runner.num_steps_per_env
                row = dict(iteration=self.iteration, total_steps=self.total_steps,
                    mean_training_reward=float(batch["environment_rewards"].mean()), failures=failures,
                    timeouts=timeouts, learning_rate=self.alg.learning_rate, **metrics)
                row.update(self.rollout_reward_metrics)
                row["action_clipping_fraction"] = self.rollout_action_clipping
                row["policy_std_max"] = float(self.model.action_std.detach().max())
                row.update(self.rollout_acceleration_peaks)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if stream.tell() == 0:
                        writer.writeheader()
                writer.writerow(row)
                stream.flush()
                if self.iteration == 1 or self.iteration % 10 == 0:
                    print("{} iter={} steps={} reward={:.4f} failures={} v/c/s={:.4f}/{:.4f}/{:.4f}".format(
                        self.cfg.ablation, self.iteration, self.total_steps,
                        row["mean_training_reward"], failures, metrics["velocity_loss"],
                        metrics["contact_loss"], metrics["slip_loss"]))
                if self.iteration % self.cfg.runner.save_interval == 0:
                    self.save()
            print("Saved {}".format(self.save()))
