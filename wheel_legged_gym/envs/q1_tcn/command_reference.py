"""Command-derived wheel feedforward and bounded wheel output helpers."""
import math


def extend_braking_wheels(mean, logits, commands, mean_limit, settings, gravity_x=None):
    """Open near-saturated wheel demands, with separate optional motion cap.

    Small corrections stay exact. Motion is unchanged unless moving_limit is set.
    No sensed or estimated body velocity enters this output parameterization.
    """
    if not settings:
        return mean
    wheels = [2, 5]
    threshold = math.atanh(settings['start']/mean_limit)
    raw = logits[:, wheels]
    extra = (raw.abs()-threshold).clamp_min(0)*settings['slope']*raw.sign()
    stationary = commands[:, :2].abs().amax(dim=-1) < 1e-6
    limits = mean.new_full((mean.shape[0], 1), settings['limit'])
    eligible = stationary
    if 'moving_limit' in settings:
        limits[~stationary] = settings['moving_limit']
        eligible = stationary.new_ones(stationary.shape)
        if 'moving_gravity_start' in settings:
            if gravity_x is None:
                raise ValueError('Motion recovery extension requires filtered IMU gravity')
            fraction = ((gravity_x.abs()-settings['moving_gravity_start']) /
                        (settings['moving_gravity_full']-settings['moving_gravity_start'])).clamp(0, 1)
            aligned = gravity_x[:, None]*raw > 0
            recovery = fraction[:, None]*aligned
            scale = recovery.where(~stationary[:, None], recovery.new_ones(recovery.shape))
            extra = extra*scale
    eligible = eligible[:, None] & (mean[:, wheels].abs() >= settings['start'])
    extended = (mean[:, wheels]+extra).maximum(-limits).minimum(limits)
    result = mean.clone()
    result[:, wheels] = mean[:, wheels].where(~eligible, extended)
    return result


def command_reference_inputs(obs, history, limit, command_scale, wheel_radius,
                             wheel_action_scale, wheel_velocity_obs_scale,
                             increment_scale=None, dt=.01, linear_velocity_scale=2.,
                             action_limit=None, feedforward_gain=1.):
    """Shift measured wheel motion and action history to a bounded command frame.

    Inputs stay untouched. The returned offset comes only from desired vx.
    Actual requested commands used for reward/evaluation remain unchanged.
    """
    if min(limit,command_scale,wheel_radius,wheel_action_scale,wheel_velocity_obs_scale,feedforward_gain)<=0:
        raise ValueError('Reference and normalization scales must be positive')

    def shift(value):
        value=value.clone()
        desired=value[...,6]/command_scale
        reference=desired.clamp(-limit,limit)
        offset=desired-reference
        value[...,6]=reference*command_scale
        angular=offset/wheel_radius
        for index in [17,20]:value[...,index]-=angular*wheel_velocity_obs_scale
        for index in [23,26]:value[...,index]-=feedforward_gain*angular/wheel_action_scale
        if increment_scale is not None:
            for index in [11,14]:value[...,index]-=angular*dt/increment_scale
        return value,offset

    current,offset=shift(obs)
    frames=history.reshape(obs.shape[0],-1,obs.shape[-1])
    shifted,_=shift(frames)
    return current,shifted.reshape_as(history),feedforward_gain*offset/(wheel_radius*wheel_action_scale)
