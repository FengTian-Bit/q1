"""T0's distance curriculum, with Q1's translational-command mapping.

Adapted from Dreamwaq-highplatform's
``legged_gym/envs/t0/mdp/curriculums.py::terrain_levels``.  Q1 commands are
``[vx, yaw_rate, height]`` rather than T0's ``[vx, vy, yaw_rate, ...]``;
only ``vx`` therefore belongs in the expected translational distance.
"""

import torch


def terrain_levels(env, env_ids):
    """Move each completed environment through T0's pre-generated map.

    This uses the final planar displacement, not accumulated path length or
    CTBC's five-episode climbing-success window.  The top row wraps to a
    random row, exactly as in T0.  Optional refresh hooks update the selected
    tile's geometry and CTBC teaching metadata after its origin changes.
    """
    if not env.init_done or len(env_ids) == 0:
        return

    distance = torch.norm(
        env.root_states[env_ids, :2] - env.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > env.terrain.env_length / 2
    expected_distance = (
        env.commands[env_ids, 0].abs() * env.max_episode_length_s * 0.5
    )
    move_down = (distance < expected_distance) & ~move_up

    env.success_ids = env_ids[move_up]
    env.fail_ids = env_ids[move_down]
    env.terrain_levels[env_ids] += move_up.long() - move_down.long()
    env.terrain_levels[env_ids] = torch.where(
        env.terrain_levels[env_ids] >= env.max_terrain_level,
        torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
        torch.clip(env.terrain_levels[env_ids], 0),
    )
    env.env_origins[env_ids] = env.terrain_origins[
        env.terrain_levels[env_ids], env.terrain_types[env_ids]
    ]

    for name in ("_refresh_terrain_classes", "_refresh_climb_targets"):
        refresh = getattr(env, name, None)
        if refresh is not None:
            refresh()
