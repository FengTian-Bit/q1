"""Q1 left/right policy consistency, following the four-wheel DreamWaQ loss.

Q1 uses +Y axes for all six joints: sagittal reflection swaps the two legs
without negating their joint angles, joint velocities or actions.
"""

import torch


JOINT_MIRROR = (3, 4, 5, 0, 1, 2)


def mirror_actions(actions):
    if actions.shape[-1] != 6:
        raise ValueError('Q1 action symmetry requires six actions')
    return actions[..., list(JOINT_MIRROR)]


def mirror_observations(observations):
    width = observations.shape[-1]
    if width not in (27, 30):
        raise ValueError('Q1 observation symmetry requires 27 or 30 columns')
    indices = list(range(width))
    for start in (9, 15, 21):  # position error, velocity, previous action
        indices[start:start+6] = [start+index for index in JOINT_MIRROR]
    signs = observations.new_ones(width)
    # Gyro is axial (-wx, wy, -wz); gravity is polar (gx, -gy, gz).
    signs[[0, 2, 4, 7]] = -1  # yaw command is column7
    if width == 30:
        signs[28] = -1  # body-frame lateral parking error; lock stays unchanged
    return observations[..., indices]*signs


def mirror_history(history, num_obs):
    if history.ndim != 2 or history.shape[-1] % num_obs:
        raise ValueError('History must contain complete flattened Q1 frames')
    frames = history.reshape(history.shape[0], -1, num_obs)
    return mirror_observations(frames).reshape_as(history)


def mirror_slope(slope):
    """Ground normal XY in the body frame: x is kept, lateral y changes sign."""
    return slope*slope.new_tensor([1.0, -1.0])


def mirror_clearance(clearance):
    """Left/right wheel clearance swap sides."""
    return clearance[..., [1, 0]]


def mirror_step(step):
    """[L10, R10, L20, R20]: left/right swap within each distance."""
    return step[..., [1, 0, 3, 2]]


TERRAIN_MIRRORS = {'slope': mirror_slope, 'clearance': mirror_clearance, 'step': mirror_step}


def policy_symmetry_loss(model, observations, history, action_mean, motion_weight=None,
                         estimates=None):
    """Both mean-action branches retain gradients; no sampled actions or critic.

    A fully selected frozen parking branch has zero weight, so its existing
    asymmetry is not used to train an unused moving controller. Terrain heads
    feed the actor, so they must be mirror-equivariant as well.
    """
    mirrored, mirrored_estimates = model.distribution(mirror_observations(observations),
                                      mirror_history(history, observations.shape[-1]))
    errors = (action_mean-mirror_actions(mirrored.mean)).square().mean(dim=-1)
    if estimates is not None:
        for name, mirror in TERRAIN_MIRRORS.items():
            if name in estimates:
                errors = errors+(estimates[name][:len(errors)]
                                 - mirror(mirrored_estimates[name])).square().mean(dim=-1)
    if motion_weight is None:
        return errors.mean()
    weight = motion_weight.detach().clamp(0, 1)
    return (errors*weight).sum()/weight.sum().clamp_min(1.)
