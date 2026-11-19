"""Causal history encoder, fixed auxiliary slots and asymmetric actor-critic."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .q1_tcn_config import ABLATIONS
from .command_reference import command_reference_inputs, extend_braking_wheels


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.ELU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def initial_action_std(value, num_actions, maximum):
    """Scalar legacy noise or one standard deviation per executed action."""
    if maximum is not None and (not math.isfinite(maximum) or maximum<=0):
        raise ValueError('max_noise_std must be positive and finite')
    std=torch.as_tensor(value,dtype=torch.get_default_dtype())
    if std.ndim==0:
        std=std.repeat(num_actions)
    if (std.shape!=(num_actions,) or not torch.isfinite(std).all()
            or (std<=0).any() or (maximum is not None and (std>maximum).any())):
        raise ValueError('init_noise_std must be positive scalar or one bounded std per action')
    return std


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
                 critic_hidden_dims=(256, 128, 64), init_noise_std=0.5,
                 mean_limit=None, max_noise_std=math.exp(2), learn_noise=True,
                 action_output_scale=None, command_reference=None,
                 wheel_residual=False, wheel_residual_limit=0.3,
                 parking_policy=False, parking_blend=False,
                 freeze_motion_backbone=True, parking_settle=None,
                 parking_fractional_flag=False, braking_wheel_extension=None,
                 wheel_residual_yaw_scale=0., steering_feedforward=None,
                 motion_wheel_gain=None, wheel_mean_scale=1.,
                 actor_output_init_gain=None, min_noise_std=math.exp(-5),
                 slow_branch=False, slow_history_length=61, slow_channels=32,
                 slow_dilations=(4, 8, 16, 32), slow_latent_dim=16,
                 slope_head=False, clearance_head=False, step_head=False):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError("Unknown ablation: {}".format(ablation))
        if not dilations or min(dilations) < 1 or kernel_size < 2:
            raise ValueError("TCN requires positive dilations and kernel_size >= 2")
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        # The fast branch always sees exactly history_length frames (the last
        # ones when a longer slow history is stored), so this check is unchanged.
        if self.receptive_field < history_length:
            raise ValueError("TCN receptive field must cover the history window")
        self.num_obs = num_obs
        self.history_length = history_length
        self.ablation = ablation
        self.enabled = ABLATIONS[ablation]
        self.command_reference = command_reference or {}
        self.wheel_residual_yaw_scale = wheel_residual_yaw_scale
        self.steering_feedforward = steering_feedforward or {}
        self.motion_wheel_gain = motion_wheel_gain or {}
        if not math.isfinite(wheel_mean_scale) or wheel_mean_scale<=0:
            raise ValueError('Wheel mean scale must be positive and finite')
        if wheel_mean_scale!=1 and (num_actions!=6 or self.command_reference
                or self.motion_wheel_gain or braking_wheel_extension):
            raise ValueError('Direct wheel mean scaling requires six actions without reference or wheel extensions')
        if actor_output_init_gain is not None and (not math.isfinite(actor_output_init_gain)
                or not 0<actor_output_init_gain<=1):
            raise ValueError('Actor output initialization gain must be in (0, 1]')
        self.wheel_mean_scale = wheel_mean_scale
        if wheel_residual_yaw_scale < 0:
            raise ValueError('Yaw residual activation scale must be nonnegative')
        if self.command_reference and (num_actions != 6 or num_obs not in (27, 30)):
            raise ValueError('Command reference requires the Q1 six-action observation layout')
        if mean_limit is not None and not 0 < mean_limit <= 1:
            raise ValueError("mean_limit must be in (0, 1] or None")
        initial_std=initial_action_std(init_noise_std,num_actions,max_noise_std)
        if min_noise_std is not None and (not math.isfinite(min_noise_std) or min_noise_std<=0):
            raise ValueError('min_noise_std must be positive and finite or None')
        if min_noise_std is not None and max_noise_std is not None and min_noise_std>max_noise_std:
            raise ValueError('min_noise_std must not exceed max_noise_std')
        self.mean_limit = mean_limit
        self.braking_wheel_extension = braking_wheel_extension or {}
        if self.braking_wheel_extension:
            cfg = self.braking_wheel_extension
            required = {'start', 'slope', 'limit'}
            optional = {'moving_limit', 'moving_gravity_start', 'moving_gravity_full'}
            if not required <= set(cfg) or set(cfg)-required-optional or mean_limit is None or num_actions != 6:
                raise ValueError('Braking wheel extension requires bounded six-action policy and start/slope/limit')
            if not 0 < cfg['start'] < mean_limit < cfg['limit'] or cfg['slope'] <= 0:
                raise ValueError('Braking extension must preserve the original small-action region')
            if 'moving_limit' in cfg and cfg['moving_limit'] <= mean_limit:
                raise ValueError('Moving wheel extension must exceed the original mean limit')
            gravity_keys = {'moving_gravity_start', 'moving_gravity_full'}
            if gravity_keys & set(cfg):
                if not gravity_keys <= set(cfg) or 'moving_limit' not in cfg:
                    raise ValueError('Motion recovery needs both IMU thresholds and a moving limit')
                if not 0 < cfg['moving_gravity_start'] < cfg['moving_gravity_full'] <= 1:
                    raise ValueError('Motion recovery IMU thresholds must lie in (0, 1]')
        self.min_log_std = math.log(min_noise_std) if min_noise_std is not None else None
        self.max_log_std = math.log(max_noise_std) if max_noise_std is not None else None
        self.register_buffer('_action_output_scale',torch.ones(num_actions),persistent=False)
        if action_output_scale is not None:
            self.set_action_output_scale(action_output_scale)
        if self.braking_wheel_extension and not torch.equal(self._action_output_scale, torch.ones_like(self._action_output_scale)):
            raise ValueError('Braking extension currently requires unchanged action units')
        self.input_projection = nn.Conv1d(num_obs, channels, 1)
        self.tcn = nn.Sequential(*[
            CausalBlock(channels, kernel_size, dilation) for dilation in dilations
        ])
        self.latent_head = nn.Linear(channels, latent_dim)
        self.velocity_head = nn.Linear(channels, 3)
        self.contact_head = nn.Linear(channels, 2)
        self.slip_head = nn.Linear(channels, 2)
        # Stage 1 terrain additions. Disabled by default: no extra parameters or
        # actor inputs, so existing checkpoints and v46 keep their exact layout.
        self.slow_branch = None
        self.slope_head = None
        self.clearance_head = None
        self.step_head = None
        extra_inputs = 0
        if slow_branch:
            if slow_history_length < history_length:
                raise ValueError('Slow branch history must cover the fast branch window')
            slow_field = 1 + (kernel_size - 1) * sum(slow_dilations)
            if slow_field < slow_history_length:
                raise ValueError('Slow TCN receptive field must cover the slow history window')
            self.slow_projection = nn.Conv1d(num_obs, slow_channels, 1)
            self.slow_branch = nn.Sequential(*[
                CausalBlock(slow_channels, kernel_size, dilation) for dilation in slow_dilations
            ])
            self.slow_latent_head = nn.Linear(slow_channels, slow_latent_dim)
            extra_inputs += slow_latent_dim
        # The terrain heads read the longest available window: slow if present.
        head_width = slow_channels if slow_branch else channels
        if slope_head:
            self.slope_head = nn.Linear(head_width, 2)
            extra_inputs += 2
        if clearance_head:
            self.clearance_head = nn.Linear(head_width, 2)
            extra_inputs += 2
        if step_head:
            # Rise of the terrain 0.10 m and 0.20 m ahead of each wheel: L10, R10, L20, R20.
            self.step_head = nn.Linear(head_width, 4)
            extra_inputs += 4
        self.slow_history_length = slow_history_length if slow_branch else history_length
        # All ablations retain seven input slots and exactly the same parameters.
        self.actor = build_mlp(num_obs + latent_dim + 7 + extra_inputs, actor_hidden_dims, num_actions)
        if actor_output_init_gain is not None:
            # A fresh policy starts close to nominal leg targets and zero wheel speed.
            # Default None preserves historical initialization exactly.
            with torch.no_grad():
                self.actor[-1].weight.mul_(actor_output_init_gain)
                self.actor[-1].bias.zero_()
        self.critic = build_mlp(num_critic_obs, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(initial_std.log())
        self.log_std.requires_grad_(learn_noise)
        self.wheel_residual = None
        self.wheel_residual_limit = wheel_residual_limit
        if wheel_residual:
            if learn_noise or not 0 < wheel_residual_limit <= 1 or num_actions != 6:
                raise ValueError('Wheel residual needs frozen exploration, six actions and a limit in (0, 1]')
            self.wheel_residual = build_mlp(num_obs+latent_dim+1, [32], 2)
            nn.init.zeros_(self.wheel_residual[-1].weight)
            nn.init.zeros_(self.wheel_residual[-1].bias)
            # Preserve the warm-start balance controller; learn moving-wheel
            # corrections and the independent privileged value function only.
            if freeze_motion_backbone:
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
            for module in (self.critic, self.wheel_residual):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.parking_policy = None
        self.parking_blend = parking_blend
        self.parking_fractional_flag = parking_fractional_flag
        if parking_fractional_flag and (not parking_blend or num_obs != 30):
            raise ValueError('Fractional parking flags require 30-slot blended parking control')
        self.parking_settle = parking_settle or {}
        if self.parking_settle:
            if not parking_blend or set(self.parking_settle) != {'gyro_limit_scaled','wheel_limit_scaled','gravity_xy_limit'}:
                raise ValueError('Parking settle filter requires blend and three normalized sensor limits')
            if min(self.parking_settle.values()) <= 0:
                raise ValueError('Parking settle sensor limits must be positive')
        if parking_blend and (not parking_policy or num_obs != 30):
            raise ValueError('Parking blend requires the frozen branch and 30-slot odometry observations')
        if parking_policy:
            if ablation != 'A0':
                raise ValueError('Frozen parking controller currently supports A0 only')
            self.parking_policy = ActorCriticTCN(
                num_obs, num_critic_obs, num_actions, ablation=ablation,
                history_length=history_length, channels=channels, dilations=dilations,
                kernel_size=kernel_size, latent_dim=latent_dim,
                actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims,
                init_noise_std=init_noise_std, mean_limit=mean_limit,
                max_noise_std=max_noise_std, min_noise_std=min_noise_std, learn_noise=False)
            for parameter in self.parking_policy.parameters():
                parameter.requires_grad_(False)

    @property
    def action_std(self):
        log_std = self.log_std
        if self.min_log_std is not None or self.max_log_std is not None:
            log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return log_std.exp()*self._action_output_scale

    def set_action_output_scale(self, values):
        scale=torch.as_tensor(values,dtype=self._action_output_scale.dtype,
                              device=self._action_output_scale.device)
        if scale.shape != self._action_output_scale.shape or not torch.isfinite(scale).all():
            raise ValueError('action_output_scale must have one finite value per action')
        if (scale<=0).any() or (scale>1).any():
            raise ValueError('action_output_scale values must be in (0, 1]')
        self._action_output_scale.copy_(scale)

    def constrain_distribution_parameters(self):
        if self.min_log_std is not None or self.max_log_std is not None:
            with torch.no_grad():
                self.log_std.clamp_(self.min_log_std, self.max_log_std)

    def encode(self, history, slow_history=None):
        history = history.reshape(-1, self.history_length, self.num_obs).transpose(1, 2)
        h = self.tcn(self.input_projection(history))[:, :, -1]
        estimates = {
            "latent": self.latent_head(h),
            # Keep checkpoint parameter layout, but do not run a disabled head.
            "velocity": self.velocity_head(h) if self.enabled[0] else h.new_zeros(h.shape[0], 3),
            "contact_logits": self.contact_head(h),
            "slip": self.slip_head(h),
        }
        terrain = h
        if self.slow_branch is not None:
            frames = slow_history.reshape(-1, self.slow_history_length, self.num_obs).transpose(1, 2)
            terrain = self.slow_branch(self.slow_projection(frames))[:, :, -1]
            estimates["slow_latent"] = self.slow_latent_head(terrain)
        if self.slope_head is not None:
            estimates["slope"] = self.slope_head(terrain)
        if self.clearance_head is not None:
            estimates["clearance"] = self.clearance_head(terrain)
        if self.step_head is not None:
            estimates["step"] = self.step_head(terrain)
        return estimates

    def split_history(self, history):
        """Return (fast 16-frame history, full slow history) from the stored window."""
        if self.slow_history_length == self.history_length:
            return history, history
        return history[:, -self.history_length*self.num_obs:], history

    def distribution(self, obs, history):
        history, slow_history = self.split_history(history)
        original_obs, original_history = obs, history
        if self.parking_fractional_flag:
            obs, history = obs.clone(), history.clone()
            obs[:, 29] = (obs[:, 29] > 0).to(obs.dtype)
            frames = history.reshape(-1, self.history_length, self.num_obs)
            frames[:, :, 29] = (frames[:, :, 29] > 0).to(frames.dtype)
        requested_command = obs[:, 6:7]
        offset = None
        if self.command_reference:
            obs, history, offset = command_reference_inputs(obs, history, **self.command_reference)
        estimates = self.encode(history, slow_history)
        slots = [estimates["velocity"], estimates["contact_logits"].sigmoid(), estimates["slip"]]
        slots = [value.detach() if enabled else torch.zeros_like(value)
                 for value, enabled in zip(slots, self.enabled)]
        # Slow latent trains through PPO like the fast latent; supervised terrain
        # heads are detached so only their own losses shape them.
        terrain = [estimates[name] if name == "slow_latent" else estimates[name].detach()
                   for name in ("slow_latent", "slope", "clearance", "step") if name in estimates]
        mean = self.actor(torch.cat([obs, estimates["latent"]] + slots + terrain, dim=-1))
        estimates["action_mean_unbounded"] = mean
        if self.mean_limit is not None:
            mean = self.mean_limit*torch.tanh(mean)
        mean = mean*self._action_output_scale
        if self.wheel_mean_scale!=1:
            mean=mean.clone();mean[:,[2,5]]=mean[:,[2,5]]*self.wheel_mean_scale
        if self.motion_wheel_gain:
            cfg=self.motion_wheel_gain
            speed=original_obs[:,6].abs()/cfg['command_scale']
            fraction=((speed-cfg['start_speed'])/(cfg['full_speed']-cfg['start_speed'])).clamp(0,1)
            gain=1+(cfg['gain']-1)*fraction
            mean=mean.clone();mean[:,[2,5]]=mean[:,[2,5]]*gain[:,None]
        if self.wheel_residual is not None:
            features = torch.cat((obs, estimates['latent'], requested_command), dim=-1)
            residual = self.wheel_residual_limit*torch.tanh(self.wheel_residual(features))
            # Exact zero correction at a zero command, including after updates.
            activation = requested_command.abs()
            if self.wheel_residual_yaw_scale:
                activation = torch.maximum(activation, original_obs[:,7:8].abs()*self.wheel_residual_yaw_scale)
            residual = residual*activation.clamp(max=1)
            mean = mean.clone()
            mean[:, [2, 5]] = mean[:, [2, 5]]+residual
            estimates['wheel_residual'] = residual
        if self.steering_feedforward:
            cfg=self.steering_feedforward
            yaw=original_obs[:,7]/cfg['command_scale']
            wheel_y=mean.new_tensor(cfg['wheel_y'])
            steering=-yaw[:,None]*wheel_y[None,:]*cfg['gain']/(cfg['wheel_radius']*cfg['wheel_action_scale'])
            mean=mean.clone();mean[:,[2,5]]=mean[:,[2,5]]+steering
        if offset is not None:
            mean = mean.clone()
            mean[:, [2, 5]] = mean[:, [2, 5]] + offset[:, None]
            bound = self.command_reference.get('action_limit', self.mean_limit)
            if bound is not None:
                mean[:, [2, 5]] = mean[:, [2, 5]].clamp(-bound, bound)
        gravity_x = None
        if 'moving_gravity_start' in self.braking_wheel_extension:
            gravity_x = original_history.reshape(-1, self.history_length, self.num_obs)[:, :, 3].mean(-1)
        mean = extend_braking_wheels(mean, estimates['action_mean_unbounded'],
                                    original_obs[:, 6:8], self.mean_limit, self.braking_wheel_extension, gravity_x)
        std = self.action_std.expand_as(mean)
        if self.parking_policy is not None:
            estimates['motion_mean'] = mean
            stationary = original_obs[:, 6:8].abs().amax(dim=-1) < 1e-6
            if self.parking_blend:
                frames = original_history.reshape(-1, self.history_length, self.num_obs)
                zero_history = frames[:, :, 6:8].abs().amax(dim=-1) < 1e-6
                eligible = frames[:, :, 29].clamp(0, 1)*zero_history
                if self.parking_settle:
                    limits = self.parking_settle
                    ready = frames[:, :, :3].abs().amax(dim=-1) <= limits['gyro_limit_scaled']
                    ready &= frames[:, :, [17, 20]].abs().amax(dim=-1) <= limits['wheel_limit_scaled']
                    ready &= frames[:, :, 3:5].norm(dim=-1) <= limits['gravity_xy_limit']
                    eligible = eligible*ready
                weight = eligible.mean(-1)
                # During braking, keep the moving balance controller until
                # the existing command-timed parking lock is active (0.5s).
                lock = original_obs[:, 29].clamp(0, 1)
                if self.parking_fractional_flag:
                    lock = (lock > 0).to(weight.dtype)
                weight = torch.where(stationary, weight*lock, weight)
            else:
                weight = stationary.to(mean.dtype)
            selected = weight > 0
            estimates['parking_weight'] = weight
            estimates['parking_mean'] = torch.zeros_like(mean)
            if selected.any():
                parking_obs = original_obs[selected].clone()
                parking_history = original_history[selected].clone()
                if self.parking_fractional_flag:
                    parking_obs[:, 29] = (parking_obs[:, 29] > 0).to(parking_obs.dtype)
                    flags = parking_history.reshape(-1, self.history_length, self.num_obs)[:, :, 29]
                    flags.copy_((flags > 0).to(flags.dtype))
                if self.parking_blend:
                    parking_obs[:, 6:8] = 0
                    parking_history.reshape(-1, self.history_length, self.num_obs)[:, :, 6:8] = 0
                parking, parking_estimates = self.parking_policy.distribution(
                    parking_obs, parking_history)
                parking_mean = extend_braking_wheels(parking.mean, parking_estimates['action_mean_unbounded'],
                    original_obs[selected, 6:8], self.parking_policy.mean_limit, self.braking_wheel_extension,
                    gravity_x[selected] if gravity_x is not None else None)
                estimates['parking_mean'][selected] = parking_mean
                mean, std = mean.clone(), std.clone()
                fraction = weight[selected, None]
                mean[selected] = fraction*parking_mean+(1-fraction)*mean[selected]
                std[selected] = fraction*parking.stddev+(1-fraction)*std[selected]
        return Normal(mean, std), estimates

    def evaluate(self, critic_obs):
        if self.command_reference or self.parking_fractional_flag:
            critic_obs = critic_obs.clone()
        if self.parking_fractional_flag:
            critic_obs[:, 32] = (critic_obs[:, 32] > 0).to(critic_obs.dtype)
        if self.command_reference:
            current = critic_obs[:, 3:3+self.num_obs]
            transformed, _, offset = command_reference_inputs(
                current, current, **self.command_reference)
            critic_obs[:, 3:3+self.num_obs] = transformed
            # Privileged velocity stays in Critic only. Match its reference frame.
            cfg = self.command_reference
            speed = offset*cfg['wheel_radius']*cfg['wheel_action_scale']/cfg.get('feedforward_gain',1.)
            critic_obs[:, 0] -= speed*cfg.get('linear_velocity_scale', 2.)
            for start in (3+self.num_obs, 3+self.num_obs+6):
                critic_obs[:, [start+2, start+5]] -= offset[:, None]
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
    # Stage 1 terrain labels, columns 10:12 slope normal XY and 12:14 wheel clearance.
    if "slope" in estimates and targets.shape[1] >= 14 and valid.any():
        losses["slope"] = F.smooth_l1_loss(estimates["slope"][valid], targets[valid, 10:12])
    if "clearance" in estimates and targets.shape[1] >= 14 and valid.any():
        losses["clearance"] = F.smooth_l1_loss(estimates["clearance"][valid], targets[valid, 12:14])
    if "step" in estimates and targets.shape[1] >= 18 and valid.any():
        losses["step"] = F.smooth_l1_loss(estimates["step"][valid], targets[valid, 14:18])
    return losses
