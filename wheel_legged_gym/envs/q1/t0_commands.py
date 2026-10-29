"""T0 command sampling and heading control mapped to Q1's three commands.

The source T0 task uses ``[vx, vy, yaw_rate, heading]``.  Q1 keeps its policy
interface ``[vx, yaw_rate, height]`` and stores heading targets separately.
Only lateral translation is omitted from the T0 command distribution.
"""

import math

import torch


def _sample_range(env, name, env_ids):
    bounds = env.command_ranges[name]
    if isinstance(bounds, torch.Tensor) and bounds.ndim == 2:
        low, high = bounds[env_ids, 0], bounds[env_ids, 1]
    else:
        low, high = bounds
    return low + (high - low) * torch.rand(len(env_ids), device=env.device)


def _sample_bounds(env, bounds, env_ids):
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
        len(env_ids), device=env.device
    )


def _heading_buffer(env):
    if not hasattr(env, "command_headings"):
        env.command_headings = torch.zeros_like(env.commands[:, 0])
    return env.command_headings


def resample_commands(env, env_ids):
    """Uniformly sample T0 commands, omitting Y and retaining Q1 height.

    T0 has no explicit standing, straight-driving or endpoint probabilities.
    With Y disabled, its 0.2 m/s planar deadband becomes ``abs(vx) <= 0.2``.
    High-platform overrides and terrain-specific speed scaling retain T0's
    ordering after that deadband, when those optional settings are present.
    """
    if getattr(env, "external_commands", False) or len(env_ids) == 0:
        return
    cfg = env.cfg.commands
    env.commands[env_ids, 0] = _sample_range(env, "lin_vel_x", env_ids)
    if cfg.heading_command:
        _heading_buffer(env)[env_ids] = _sample_range(env, "heading", env_ids)
    else:
        env.commands[env_ids, 1] = _sample_range(env, "ang_vel_yaw", env_ids)
    env.commands[env_ids, 0] *= env.commands[env_ids, 0].abs() > 0.2
    env.commands[env_ids, 2] = cfg.ranges.height[0]

    terrain_types = getattr(env, "env_class", None)
    high_cfg = getattr(cfg, "highplatform", None)
    if terrain_types is not None and high_cfg is not None:
        high_type = int(getattr(env.cfg.terrain, "highplatform_terrain_type", 5))
        high_ids = env_ids[terrain_types[env_ids].long() == high_type]
        if len(high_ids):
            env.commands[high_ids, 0] = _sample_bounds(env, high_cfg.lin_vel_x, high_ids)
            if cfg.heading_command:
                _heading_buffer(env)[high_ids] = _sample_bounds(env, high_cfg.heading, high_ids)

    scale_cfg = getattr(cfg, "terrain_speed_scale", None)
    if terrain_types is not None and scale_cfg is not None:
        scaled_types = torch.as_tensor(
            scale_cfg.terrain_types, device=env.device, dtype=torch.long
        )
        mask = (terrain_types[env_ids].long()[:, None] == scaled_types).any(dim=1)
        env.commands[env_ids[mask], 0] *= scale_cfg.scale

    if hasattr(env, "command_targets"):
        env.command_targets[env_ids] = env.commands[env_ids]
    if hasattr(env, "needs_command_sample"):
        env.needs_command_sample[env_ids] = False
    if hasattr(env, "command_sample_step"):
        env.command_sample_step[env_ids] = env.episode_length_buf[env_ids]


def update_heading_command(env):
    """Apply T0's 0.5 heading gain and +/-1 rad/s saturation every step."""
    if (getattr(env, "external_commands", False)
            or not env.cfg.commands.heading_command):
        return
    quat = env.base_quat
    # XY of the rotated forward axis, matching quat_apply(q, [1, 0, 0]).
    heading = torch.atan2(
        2 * (quat[:, 0] * quat[:, 1] + quat[:, 3] * quat[:, 2]),
        1 - 2 * (quat[:, 1].square() + quat[:, 2].square()),
    )
    error = torch.remainder(_heading_buffer(env) - heading, 2 * math.pi)
    # T0's wrap_to_pi keeps +pi, rather than mapping it to -pi.
    error = torch.where(error > math.pi, error - 2 * math.pi, error)
    yaw = (0.5 * error).clamp(-1.0, 1.0)
    direct = getattr(env, "flat_direct_command", None)
    if direct is not None:
        yaw = torch.where(direct, env.commands[:, 1], yaw)
    env.commands[:, 1] = yaw
    if hasattr(env, "command_targets"):
        env.command_targets[:, 1] = yaw


def compute_commands(env):
    """Run T0's periodic sampling followed by its per-step heading update."""
    if getattr(env, "external_commands", False):
        if hasattr(env, "command_targets"):
            env.commands[:] = env.command_targets
        return
    interval = int(env.cfg.commands.resampling_time / env.dt)
    if interval > 0:
        env_ids = (env.episode_length_buf % interval == 0).nonzero(
            as_tuple=False
        ).flatten()
        resample_commands(env, env_ids)
    update_heading_command(env)
