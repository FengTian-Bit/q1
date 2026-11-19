"""Preserve policy inputs when changing normalized wheel action units."""


def remap_wheel_action_history(state, num_obs, velocity_scale_ratio):
    """Convert checkpoint inputs for new_scale/old_scale, leaving outputs alone.

    Physical wheel targets stay the same when new normalized wheel actions
    equal old actions / ratio. Compensate every observed previous-action slot
    in the TCN, Actor and privileged Critic. Output mean/std must separately
    be scaled by 1/ratio during transfer. Does not mutate the source state.
    """
    if num_obs < 27 or velocity_scale_ratio <= 0:
        raise ValueError('Expected Q1 proprioception and a positive scale ratio')
    converted = {name: value.clone() for name, value in state.items()}
    observation_columns = [23, 26]  # previous action: L/R wheel
    converted['input_projection.weight'][:, observation_columns, :] *= velocity_scale_ratio
    converted['actor.0.weight'][:, observation_columns] *= velocity_scale_ratio
    # Critic = velocity[3], current obs, previous actions[6] twice, ...
    critic_columns = [3 + index for index in observation_columns]
    for offset in [3 + num_obs, 3 + num_obs + 6]:
        critic_columns.extend([offset + 2, offset + 5])
    converted['critic.0.weight'][:, critic_columns] *= velocity_scale_ratio
    return converted
