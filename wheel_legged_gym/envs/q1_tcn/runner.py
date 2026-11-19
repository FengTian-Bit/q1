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
    # A complete old reward snapshot keeps its historical caps and weights.
    # Partial overrides use today's defaults; new snapshots save the switch.
    rewards = values.get('rewards', {})
    if (hasattr(obj, 'rewards') and hasattr(obj.rewards, 'clip_constraint_rewards')
            and {'scales', 'clip_single_reward'} <= set(rewards)
            and 'clip_constraint_rewards' not in rewards):
        values = deepcopy(values)
        values['rewards']['clip_constraint_rewards'] = True
        for name in ['stand_still', 'stand_wheel_still', 'run_still']:
            values['rewards']['scales'].setdefault(name, 0.0)
    policy = values.get('policy', {})
    if (hasattr(obj, 'policy') and hasattr(obj.policy, 'min_noise_std')
            and {'init_noise_std', 'max_noise_std', 'history_length'} <= set(policy)
            and 'min_noise_std' not in policy):
        values = deepcopy(values)
        values['policy']['min_noise_std'] = float(np.exp(-5))
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


class EpisodeRewardAccumulator:
    """Aggregate completed episodes, weighted by reset count rather than steps."""
    def __init__(self, names):
        self.totals = {name: 0.0 for name in names}
        self.count = 0
        self.length_sum = 0.0

    def add(self, episode_info, dones, episode_lengths):
        count = int(dones.sum())
        if not count or not episode_info:
            return
        for name in self.totals:
            # LeggedRobot already divides by the configured maximum episode seconds.
            self.totals[name] += episode_info['rew_'+name]*count
        self.count += count
        self.length_sum += episode_lengths[dones.bool()].sum()

    def metrics(self):
        result = {'completed_episode_count': self.count,
                  'completed_episode_length_mean_s': float(self.length_sum/self.count) if self.count else None}
        result.update({'episode_reward_'+name: float(total/self.count) if self.count else None
                       for name, total in self.totals.items()})
        return result


def command_mode_moments(sample):
    """Target type and actual tracking error, captured before automatic resets."""
    x_zero = sample['target_command'] == 0
    yaw_zero = sample['yaw_command'] == 0
    masks = dict(stand=x_zero & yaw_zero, straight=~x_zero & yaw_zero,
                 pure_yaw=x_zero & ~yaw_zero, combined=~x_zero & ~yaw_zero)
    vx_error = sample['velocity'][:, 0]-sample['command']
    yaw_error = sample['yaw_rate']-sample['yaw_command']
    result = {}
    for name, mask in masks.items():
        result[name+'_count'] = mask.sum()
        result[name+'_vx_sse'] = vx_error[mask].square().sum()
        result[name+'_yaw_sse'] = yaw_error[mask].square().sum()
    return result


class TCNPpoRunner:
    def __init__(self, env, train_cfg, log_dir, device="cpu"):
        self.env, self.cfg = env, train_cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = ActorCriticTCN(env.num_obs, env.num_privileged_obs, env.num_actions,
            ablation=train_cfg.ablation, **config_dict(train_cfg.policy)).to(self.device)
        self.alg = PPO(self.model, **config_dict(train_cfg.algorithm),
                       mask_symmetry_swing=bool(getattr(env.cfg, 'ctbc', None) is not None
                                                and env.cfg.ctbc.enabled
                                                and env.cfg.ctbc.symmetry_mask))
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
        command_metrics = {}
        mode_metrics = {}
        maximum_stand_targets = 0
        stand_count=int(env.num_envs*env.cfg.commands.permanent_stand_fraction)
        parking=env.cfg.commands.parking_sequence
        parking_end=stand_count+int(env.num_envs*parking['fraction']) if parking else stand_count
        action_clipping = 0.0
        acceleration_peaks = {}
        episode_rewards = EpisodeRewardAccumulator(env.episode_sums)
        with torch.no_grad():
            for _ in range(self.cfg.runner.num_steps_per_env):
                obs, history = env.get_observations()
                obs, history = obs.to(self.device).clone(), history.to(self.device).clone()
                critic_obs = env.get_privileged_observations().to(self.device).clone()
                targets = env.estimator_targets.to(self.device).clone()
                distribution, _ = model.distribution(obs, history)
                actions = distribution.sample()
                values = model.evaluate(critic_obs)
                episode_steps = env.episode_length_buf.clone()
                _, _, rewards, dones, infos, _ = env.step(actions.to(env.device))
                episode_rewards.add(infos.get('episode', {}), dones, (episode_steps+1)*env.dt)
                if env.cfg.commands.axis_sampling:
                    moments = command_mode_moments(env.transition_metrics)
                    for name, value in moments.items():
                        mode_metrics[name] = mode_metrics.get(name, 0)+value
                    maximum_stand_targets = max(maximum_stand_targets, int(moments['stand_count']))
                for name, value in getattr(env, 'parking_tail_metrics', {}).items():
                    if name.endswith('_max'):
                        command_metrics[name] = torch.maximum(command_metrics.get(name, value), value)
                    else:
                        command_metrics[name] = command_metrics.get(name, 0.)+value
                for name, term in env.reward_terms.items():
                    reward_metrics[name] = reward_metrics.get(name, 0.0) + term.mean()
                if stand_count>0:
                    sample=env.transition_metrics
                    zero=(sample['command'].abs()<env.cfg.rewards.parking_speed_threshold)
                    zero &= sample['yaw_command'].abs()<env.cfg.rewards.parking_yaw_threshold
                    full=sample['parking_weight']>=.999
                    error=sample['velocity'][:,0]-sample['command']
                    command_values=dict(training_zero_command_fraction=zero.float().mean(),
                        training_parking_full_weight_fraction=full.float().mean(),
                        stand_group_parking_full_weight_fraction=full[:stand_count].float().mean(),
                        motion_group_parking_full_weight_fraction=(full[stand_count:].float().mean()
                            if stand_count<env.num_envs else full.new_zeros((),dtype=torch.float)),
                        stand_group_vx_mse=error[:stand_count].square().mean(),
                        motion_group_vx_mse=(error[stand_count:].square().mean()
                            if stand_count<env.num_envs else error.new_zeros(())))
                    for name,value in command_values.items():
                        command_metrics[name]=command_metrics.get(name,0.)+value
                    for name,mask in [('stand',slice(None,stand_count)),('motion',slice(stand_count,None))]:
                        key=name+'_group_failures'
                        command_metrics[key]=command_metrics.get(key,0.)+sample['failure'][mask].sum()
                    if parking:
                        for name,mask in [('parking_sequence',slice(stand_count,parking_end)),
                                          ('regular',slice(parking_end,None))]:
                            selected=error[mask]
                            mse=selected.square().mean() if selected.numel() else error.new_zeros(())
                            key=name+'_group_vx_mse'
                            command_metrics[key]=command_metrics.get(key,0.)+mse
                            key=name+'_group_failures'
                            command_metrics[key]=command_metrics.get(key,0.)+sample['failure'][mask].sum()
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
                    ctbc_swing=env.transition_metrics.get(
                        'ctbc_swing', torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)),
                    actions=actions, values=values, log_prob=distribution.log_prob(actions).sum(-1),
                    mean=distribution.mean, std=distribution.stddev, rewards=rewards, dones=dones,
                    environment_rewards=environment_rewards)
                for name, value in transition.items():
                    batch.setdefault(name, []).append(value.clone())
            last_values = model.evaluate(env.get_privileged_observations().to(self.device))
        steps = self.cfg.runner.num_steps_per_env
        self.rollout_reward_metrics = {"reward_"+name: float(value/steps) for name, value in reward_metrics.items()}
        self.rollout_command_metrics={}
        for name,value in command_metrics.items():
            if name.endswith('_failures'):
                self.rollout_command_metrics[name]=int(value)
            elif name.startswith('parking_tail_'):
                self.rollout_command_metrics[name] = int(value) if name.endswith('_frames') else float(value)
            elif name.endswith('_vx_mse'):
                self.rollout_command_metrics[name[:-3]+'rmse']=float((value/steps).sqrt())
            else:
                self.rollout_command_metrics[name]=float(value/steps)
        self.rollout_action_clipping = float(action_clipping/steps)
        if env.cfg.commands.axis_sampling:
            if maximum_stand_targets > int(env.num_envs*.2):
                raise RuntimeError('Parking target count exceeded the user cap of20%')
            for mode in ['stand', 'straight', 'pure_yaw', 'combined']:
                count = int(mode_metrics[mode+'_count'])
                self.rollout_command_metrics['command_target_'+mode+'_fraction'] = count/(steps*env.num_envs)
                for axis in ['vx', 'yaw']:
                    self.rollout_command_metrics[mode+'_target_'+axis+'_rmse'] = (
                        float((mode_metrics[mode+'_'+axis+'_sse']/count).sqrt()) if count else None)
            self.rollout_command_metrics['maximum_stand_target_count'] = maximum_stand_targets
        self.rollout_acceleration_peaks = {name+'_max':float(value) for name,value in acceleration_peaks.items()}
        self.rollout_episode_metrics = episode_rewards.metrics()
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
             allow_yaw_weight_update=False, allow_wheel_diff_update=False, allow_reward_weight_update=False):
        checkpoint = load_checkpoint(path, self.device)
        saved = checkpoint["config"]
        reward_weight_updates = {}
        # Resume must preserve conditions and capacity, not just matching tensor shapes.
        for section in ("env", "train"):
            previous, current = deepcopy(saved[section]), deepcopy(self.snapshot[section])
            if section == 'env':
                previous['rewards'].setdefault('clip_constraint_rewards',True)
                for name in ['stand_still','stand_wheel_still','run_still']:
                    previous['rewards']['scales'].setdefault(name,0.0)
                previous['rewards'].setdefault('relative_tracking',{})
                previous['rewards'].setdefault('multi_scale_tracking',{})
                previous['rewards'].setdefault('multi_scale_parking',{})
                previous['rewards'].setdefault('adaptive_tracking_sigma',{})
                previous['rewards'].setdefault('yaw_tracking_mixture',{})
                previous['rewards'].setdefault('adaptive_yaw_tracking_sigma',{})
                previous['rewards'].setdefault('positive_stand_rewards',{})
                previous['rewards'].setdefault('wheel_diff_tracking',{})
                previous['rewards']['scales'].setdefault('tracking_wheel_diff',0.)
                if allow_reward_weight_update:
                    old_weights=previous['rewards']['scales']
                    new_weights=current['rewards']['scales']
                    if set(old_weights)!=set(new_weights) or any(not np.isfinite(v) for v in new_weights.values()):
                        raise ValueError('Reward-only continuation requires the same finite reward weight fields')
                    reward_weight_updates={name:dict(before=old_weights[name],after=value)
                        for name,value in new_weights.items() if value!=old_weights[name]}
                    previous['rewards']['scales']=deepcopy(new_weights)
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
                previous['commands'].setdefault('permanent_stand_fraction',0.)
                previous['commands'].setdefault('parking_sequence',{})
                previous['commands'].setdefault('axis_sampling',{})
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
                previous['policy'].setdefault('min_noise_std',float(np.exp(-5)))
                previous['policy'].setdefault('wheel_mean_scale',1.0)
                previous['policy'].setdefault('actor_output_init_gain',None)
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
            if section == 'env':
                # Only exact disabled defaults are compatible with old files.
                # Any enabled replay or changed threshold still fails equality.
                previous['terrain'].setdefault('climb_replay_quarter', False)
                previous['terrain'].setdefault('climb_replay_max_riser', .165)
                previous['rewards'].setdefault('parking_huber_delta_m', 0.)
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
        return reward_weight_updates

    def learn(self, iterations):
        self.model.train()
        csv_path = self.log_dir / "training.csv"
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = None
            for _ in range(iterations):
                import time
                started = time.perf_counter()
                batch, last_values, failures, timeouts = self.collect()
                collected = time.perf_counter()
                metrics = self.alg.update(batch, last_values)
                updated = time.perf_counter()
                self.iteration += 1
                if hasattr(self.env, 'set_training_iteration'):
                    self.env.set_training_iteration(self.iteration)
                self.total_steps += self.env.num_envs*self.cfg.runner.num_steps_per_env
                row = dict(iteration=self.iteration, total_steps=self.total_steps,
                    mean_training_reward=float(batch["environment_rewards"].mean()), failures=failures,
                    timeouts=timeouts, learning_rate=self.alg.learning_rate, **metrics)
                row["rollout_wall_seconds"] = collected - started
                row["ppo_wall_seconds"] = updated - collected
                row.update(self.rollout_reward_metrics)
                row.update(self.rollout_episode_metrics)
                row.update(self.rollout_command_metrics)
                row["action_clipping_fraction"] = self.rollout_action_clipping
                row["policy_std_max"] = float(self.model.action_std.detach().max())
                row.update(self.rollout_acceleration_peaks)
                if getattr(self.env, "custom_origins", False):
                    # Stage 1: where the curriculum has put the robots, recorded every iteration.
                    levels = self.env.terrain_levels
                    rows = self.env.cfg.terrain.num_rows
                    counts = torch.bincount(levels.clamp(0, rows-1), minlength=rows).float()/levels.numel()
                    row["terrain_level_mean"] = float(levels.float().mean())
                    episodes = getattr(self.env, 'climb_episodes', 0)
                    row["climb_episodes"] = episodes
                    row["climb_success_rate"] = (self.env.climb_successes/episodes) if episodes else float('nan')
                    row["climb_fall_rate"] = (self.env.climb_falls/episodes) if episodes else float('nan')
                    if episodes:
                        self.env.climb_episodes = self.env.climb_successes = self.env.climb_falls = 0
                    if hasattr(self.env, '_climb_tile_mask'):
                        ids = torch.arange(self.env.num_envs, device=levels.device)
                        mask = self.env._climb_tile_mask(ids)
                        row["climb_tile_fraction"] = float(mask.float().mean())
                        if hasattr(self.env, 'climb_replay_mask'):
                            standing = ids < int(self.env.num_envs*self.env.cfg.commands.permanent_stand_fraction)
                            groups = getattr(self.env, 'climb_group_outcomes', {})
                            row['climb_unclassified_episodes'] = row['climb_episodes']-sum(v[0] for v in groups.values())
                            for group, membership in [('replay', self.env.climb_replay_mask),
                                                      ('progression', ~self.env.climb_replay_mask)]:
                                selected = mask & ~standing & membership
                                row['climb_'+group+'_envs'] = int(selected.sum())
                                row['climb_'+group+'_level_mean'] = float(levels[selected].float().mean()) if bool(selected.any()) else 0.
                                for key, value in zip(('episodes', 'successes', 'falls'), groups.get(group, [0, 0, 0])):
                                    row['climb_'+group+'_'+key] = value
                            self.env.climb_group_outcomes = {g: [0, 0, 0] for g in ('replay', 'progression')}
                        row["climb_level_mean"] = float(levels[mask].float().mean()) if bool(mask.any()) else float('nan')
                    row.update({"terrain_level_hist_{}".format(level): float(counts[level])
                                for level in range(rows)})
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
