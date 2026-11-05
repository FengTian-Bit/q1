"""Flat Q1 with aligned state labels and finite-duration trunk force pulses."""

import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from wheel_legged_gym.envs.q1_flat.q1_flat import Q1Flat
from .labels import flat_wheel_slip, update_contact
from .parking import ParkingState
from .odometry import WheelOdometry
from .q1_tcn_config import configure_observation_layout


def flat_sequence_velocity(time_s, magnitude, sign):
    """Stop/forward/stop/reverse/stop; no phase or clock is an Actor input."""
    phase = time_s.remainder(12.)
    forward = ((phase >= 2.) & (phase < 5.)).to(time_s.dtype)
    reverse = ((phase >= 7.) & (phase < 10.)).to(time_s.dtype)
    return (forward-reverse)*magnitude*sign


def target_sequence_commands(time_s, ids, linear_limit, yaw_limit, minimum, edge_fraction, stage):
    """Full-range commands from iteration one; straight/spin/combined families."""
    phase=time_s.remainder(16.)
    forward=((phase>=2.) & (phase<6.)).to(time_s.dtype)
    reverse=((phase>=9.) & (phase<13.)).to(time_s.dtype)
    direction=forward-reverse
    linear=linear_limit*(minimum+(1-minimum)*(ids%11).float()/10)
    yaw=yaw_limit*(minimum+(1-minimum)*((ids//3)%11).float()/10)
    edge=(ids%100).float()<100*edge_fraction
    linear=torch.where(edge,torch.full_like(linear,linear_limit),linear)
    yaw=torch.where(edge,torch.full_like(yaw,yaw_limit),yaw)
    vx=direction*linear*(1-2*(ids%2).float())
    wz=torch.zeros_like(vx)
    if stage=='turning':
        family=(ids//2)%4
        yaw_direction=torch.where(family==3,forward+reverse,direction)
        wz=yaw_direction*yaw*(1-2*((ids//8)%2).float())
        vx=torch.where(family==1,torch.zeros_like(vx),vx)
        wz=torch.where(family==0,torch.zeros_like(wz),wz)
    return vx,wz


def wide_tracking_reward(error, sigma):
    """Gaussian accuracy plus a Huber cost with a useful far-error slope."""
    scaled=error/sigma.sqrt()
    absolute=scaled.abs()
    huber=torch.where(absolute<1,.5*scaled.square(),absolute-.5)
    return torch.exp(-scaled.square())-.2*huber


def rate_limit_linear_command(previous, target, dt, acceleration):
    """Limit desired vx changes; never use a sensed or estimated body velocity."""
    if acceleration <= 0:
        return target.clone()
    increment = acceleration*dt
    return previous+(target-previous).clamp(-increment, increment)


class Q1TCN(Q1Flat):
    def _parse_cfg(self, cfg):
        configure_observation_layout(cfg)
        super()._parse_cfg(cfg)

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        props[0].com.x += self.cfg.estimator.payload_com_offset_x
        # Store the realized COM, rather than only the randomization delta.
        self.base_com[env_id] = torch.tensor(
            [props[0].com.x, props[0].com.y, props[0].com.z], device=self.device)
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, self.num_bodies, 13)
        properties = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.wheel_com_local = torch.tensor([
            [properties[i].com.x, properties[i].com.y, properties[i].com.z]
            for i in self.feet_indices.tolist()
        ], device=self.device).expand(self.num_envs, -1, -1)
        self.wheel_center_local = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.wheel_center_local[:, :, 1] = torch.tensor(
            self.cfg.estimator.wheel_center_y, device=self.device)
        self.wheel_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.contact_counter = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.estimator_targets = torch.zeros(self.num_envs, 10, device=self.device)
        self.history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.pulse_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.eval_force = torch.zeros_like(self.pulse_force)
        self.evaluation_mode = False
        self.external_commands = False
        self.transition_metrics = {}
        self.command_targets = self.commands.clone()
        self.linear_command_state = self.commands[:, 0].clone()
        self.command_acceleration = torch.zeros(self.num_envs, device=self.device)
        self.previous_world_velocity = self.root_states[:, 7:10].clone()
        self.terminal_critic_obs = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.parking = ParkingState(self.num_envs, self.device)
        self.raw_action_clipping = torch.zeros(self.num_envs, device=self.device)
        self.reward_terms = {}
        self.odometry = (WheelOdometry(self.num_envs, self.device,
            radius=self.cfg.asset.wheel_radius, l1=self.cfg.asset.l1, l2=self.cfg.asset.l2,
            gravity_gain=self.cfg.odometry.gravity_gain) if self.cfg.odometry.enabled else None)
        self.odometry_reference = torch.zeros(self.num_envs, 2, device=self.device)
        self.odometry_features = torch.zeros(self.num_envs, 3, device=self.device)
        self.sensor_cache = torch.zeros(self.num_envs, 27, device=self.device)
        self.sensor_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wheel_increment = torch.zeros(self.num_envs, 2, device=self.device)

    def compute_proprioception_observations(self):
        obs = super().compute_proprioception_observations()
        if self.cfg.odometry.observe_wheel_increments:
            obs[:, [11, 14]] = self.wheel_increment/self.cfg.odometry.increment_scale
        if self.cfg.odometry.enabled:
            return torch.cat((obs, self.odometry_features), dim=-1)
        return obs

    def _measure_proprioception(self):
        obs = super().compute_proprioception_observations()
        if self.add_noise:
            obs += (2*torch.rand_like(obs)-1)*self.noise_scale_vec[:27]
        limit = self.cfg.normalization.clip_observations
        return obs.clamp(-limit, limit)

    def _odometry_measurements(self):
        # Shared noisy IMU/leg samples also go to Actor. Wheel phase is an
        # encoder measurement used only through increments in dead reckoning.
        q = self.sensor_cache[:, 9:15]/self.obs_scales.dof_pos
        q = q + self.default_dof_pos[:, self.policy_dof_indices]
        q[:, [2, 5]] = self.dof_pos[:, self.wheel_dof_indices]
        if self.add_noise:
            noise = self.cfg.noise
            q[:, [2, 5]] += (2*torch.rand(self.num_envs, 2, device=self.device)-1)*noise.noise_level*noise.noise_scales.dof_pos
        return q, self.sensor_cache[:, :3]/self.obs_scales.ang_vel, self.sensor_cache[:, 3:6]

    def _parking_observation(self):
        active = self.parking.stationary & self.parking.locked
        active &= (self.parking.initial_stand |
                   (self.parking.elapsed >= self.cfg.odometry.parking_observation_delay_s))
        delta = self.odometry.xy-self.odometry_reference
        yaw = self.odometry.yaw
        forward = yaw.cos()*delta[:, 0]+yaw.sin()*delta[:, 1]
        lateral = -yaw.sin()*delta[:, 0]+yaw.cos()*delta[:, 1]
        error = torch.stack((forward, lateral), dim=-1)/self.cfg.odometry.position_scale
        self.odometry_features[:, :2] = error.clamp(-5, 5)*active[:, None]
        self.odometry_features[:, 2] = active.float()
        ramp = self.cfg.odometry.parking_blend_ramp_s
        if ramp > 0:
            start = max(self.cfg.rewards.parking_brake_s, self.cfg.odometry.parking_observation_delay_s)
            fraction = ((self.parking.elapsed-start)/ramp).clamp(0, 1)
            # Positive encodes the unchanged binary lock for the frozen nets;
            # magnitude controls only the handoff rate in the policy wrapper.
            fraction = fraction.clamp_min(torch.finfo(fraction.dtype).eps)
            fraction = torch.where(self.parking.initial_stand, torch.ones_like(fraction), fraction)
            self.odometry_features[:, 2] = active.float()*fraction

    def step(self, actions):
        limits = actions.new_full((self.num_actions,), self.cfg.normalization.clip_actions)
        leg_limit = self.cfg.normalization.leg_clip_actions
        if leg_limit is not None:
            limits[[0, 1, 3, 4]] = leg_limit
        self.raw_action_clipping = (actions.to(self.device).abs() > limits.to(self.device)).float().mean(-1)
        if leg_limit is not None:
            actions = actions.clone()
            actions[:, [0, 1, 3, 4]] = actions[:, [0, 1, 3, 4]].clamp(-leg_limit, leg_limit)
        return super().step(actions)

    def compute_reward(self):
        """Same dt scaling/clipping as LeggedRobot, with per-term logging."""
        self.rew_buf.zero_()
        self.reward_terms = {}
        limit = self.cfg.rewards.clip_single_reward*self.dt
        for name, function in zip(self.reward_names, self.reward_functions):
            term = (function()*self.reward_scales[name]).clamp(-limit, limit)
            self.reward_terms[name] = term
            self.rew_buf += term
            self.episode_sums[name] += term
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf.clamp_min_(0)
        if "termination" in self.reward_scales:
            term = self._reward_termination()*self.reward_scales["termination"]
            self.reward_terms["termination"] = term
            self.rew_buf += term
            self.episode_sums["termination"] += term

    def _reward_tracking_ang_vel(self):
        difference = self.commands[:, 1]-self.base_ang_vel[:, 2]
        cfg = self.cfg.rewards
        sigma = cfg.tracking_sigma if cfg.reward_version == 1 else cfg.yaw_tracking_sigma
        if cfg.reward_version>=6:
            stationary=(self.commands[:,0].abs()<cfg.parking_speed_threshold)&(self.commands[:,1].abs()<cfg.parking_yaw_threshold)
            moving_sigma=torch.full_like(difference,sigma)
            if cfg.reward_version>=7:
                relative=cfg.relative_tracking
                width=(self.commands[:,1].abs()*relative['fraction']).clamp_min(relative['yaw_floor'])
                width=torch.where(self.commands[:,1].abs()<cfg.parking_yaw_threshold,
                                  torch.full_like(width,relative['zero_yaw_width']),width)
                moving_sigma=width.square()
            dense=wide_tracking_reward(difference,moving_sigma)
            return torch.where(stationary,torch.exp(-difference.square()/sigma),dense)
        return torch.exp(-difference.square()/sigma)

    def _reward_tracking_wheel_diff(self):
        # Q1 wheel axes are both +Y; positive yaw requires right faster than left.
        cfg = self.cfg.rewards.wheel_diff_tracking
        wheel_velocity = self.dof_vel[:, self.policy_dof_indices][:, [2, 5]]
        wheel_yaw = ((wheel_velocity[:, 1]-wheel_velocity[:, 0])
                     *self.cfg.asset.wheel_radius/cfg['track_width'])
        error = self.commands[:, 1]-wheel_yaw
        turning = self.commands[:, 1].abs() >= self.cfg.rewards.parking_yaw_threshold
        supported = (self.contact_forces[:, self.feet_indices, 2]
                     > self.cfg.estimator.contact_off_force).all(dim=-1)
        return torch.exp(-error.square()/cfg['sigma'])*(turning & supported)

    def _reward_tracking_lin_vel(self):
        cfg = self.cfg.rewards
        difference = self.commands[:, 0]-self.base_lin_vel[:, 0]
        if cfg.reward_version == 1:
            return torch.exp(-difference.square()/cfg.tracking_sigma)
        stationary = self.commands[:, 0].abs() < cfg.parking_speed_threshold
        if cfg.reward_version>=6:
            stationary &= self.commands[:,1].abs()<cfg.parking_yaw_threshold
        moving_sigma=torch.full_like(difference,cfg.tracking_sigma)
        if cfg.reward_version>=7:
            relative=cfg.relative_tracking
            moving_sigma=(self.commands[:,0].abs()*relative['fraction']).clamp_min(relative['linear_floor']).square()
        sigma = torch.where(stationary, cfg.idle_tracking_sigma, moving_sigma)
        if cfg.reward_version>=6:
            return torch.where(stationary,torch.exp(-difference.square()/sigma),wide_tracking_reward(difference,sigma))
        return torch.exp(-difference.square()/sigma)

    def _reward_parking_drift(self):
        return self.parking.penalty(self.cfg.rewards)

    def _reward_parking_velocity(self):
        cfg = self.cfg.rewards
        speed = self.base_lin_vel[:, :2].norm(dim=-1)
        error = (speed-cfg.parking_velocity_deadband).clamp_min(0)/cfg.parking_velocity_scale
        return self.parking.weight*error.clamp(max=4)

    def _reward_termination(self):
        # Simultaneous timeout must not hide an actual fall in reward v2.
        if self.cfg.rewards.reward_version >= 2:
            return self.physical_failure.float()
        return super()._reward_termination()

    def _resample_commands(self, env_ids):
        if not getattr(self, "external_commands", False):
            super()._resample_commands(env_ids)
            self._apply_sequence_commands(env_ids)
            self.command_targets[env_ids] = self.commands[env_ids]

    def _advance_linear_command(self):
        previous = self.linear_command_state.clone()
        current = rate_limit_linear_command(previous, self.command_targets[:, 0], self.dt,
                                            self.cfg.commands.max_linear_acceleration)
        self.commands[:, 0] = current
        self.linear_command_state.copy_(current)
        self.command_acceleration.copy_((current-previous)/self.dt)

    def set_command_targets(self, forward_speed, yaw_rate=0.0):
        """External drivers call once before each policy step, including play."""
        self.command_targets[:, 0] = forward_speed
        self.command_targets[:, 1] = yaw_rate
        self.commands[:, 1] = yaw_rate
        self._advance_linear_command()

    def _apply_sequence_commands(self, env_ids=None):
        fraction = self.cfg.commands.sequence_fraction
        if fraction <= 0 or getattr(self, 'external_commands', False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = env_ids[env_ids < int(self.num_envs*fraction)]
        # Vary dwell times by +/-20%; cover configured amplitudes and both signs.
        pace = .8+.1*(ids % 5).float()
        time_s = self.episode_length_buf[ids].float()*self.dt/pace
        limit = min(-self.cfg.commands.ranges.lin_vel_x[0], self.cfg.commands.ranges.lin_vel_x[1])
        minimum = self.cfg.commands.sequence_min_fraction
        if self.cfg.commands.target_mode:
            yaw_limit=min(-self.cfg.commands.ranges.ang_vel_yaw[0],self.cfg.commands.ranges.ang_vel_yaw[1])
            vx,wz=target_sequence_commands(time_s,ids,limit,yaw_limit,minimum,
                self.cfg.commands.target_edge_fraction,self.cfg.commands.target_stage)
            self.commands[ids,0]=vx;self.commands[ids,1]=wz
            return
        magnitude = limit*(minimum+(1-minimum)*(ids % 11).float()/10)
        sign = 1.-2.*(ids % 2).float()
        self.commands[ids, 0] = flat_sequence_velocity(time_s, magnitude, sign)
        self.commands[ids, 1] = 0

    def _post_physics_step_callback(self):
        external = getattr(self, 'external_commands', False)
        if not external and self.cfg.commands.max_linear_acceleration > 0:
            self.commands[:] = self.command_targets
        super()._post_physics_step_callback()
        self._apply_sequence_commands()
        if not external:
            self.command_targets[:] = self.commands
            self._advance_linear_command()

    def _push_robots(self):
        """Called every physical step, including zero-force steps after a pulse."""
        cfg = self.cfg.domain_rand
        if self.evaluation_mode:
            force = self.eval_force
        else:
            period = max(1, round(cfg.push_interval_s/self.sim_params.dt))
            duration = max(1, round(cfg.push_duration_s/self.sim_params.dt))
            phase = (self.envs_steps_buf - 1).remainder(period)
            start = phase == 0
            # First pulse at 5 s, allowing a newly reset robot to settle.
            active = (phase < duration) & (self.envs_steps_buf > period)
            ids = (start & active).nonzero(as_tuple=False).flatten()
            self.pulse_force[ids, :2] = (
                2*torch.rand(len(ids), 2, device=self.device)-1)*cfg.push_force_xy
            force = self.pulse_force * active[:, None]
        self.rigid_body_external_forces.zero_()
        self.rigid_body_external_torques.zero_()
        self.rigid_body_external_forces[:, 0] = force
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques), gymapi.ENV_SPACE)

    def _critic_observations(self, obs):
        heights = (self.root_states[:, 2:3] - 0.5 - self.measured_heights).clamp(-1, 1)
        return torch.cat((
            self.base_lin_vel*self.obs_scales.lin_vel, obs,
            self.last_actions[:, :, 0], self.last_actions[:, :, 1],
            self.dof_acc*self.obs_scales.dof_acc, heights*self.obs_scales.height_measurements,
            self.torques*self.obs_scales.torque, (self.base_mass-self.base_mass.mean()).view(-1, 1),
            self.base_com, self.default_dof_pos-self.raw_default_dof_pos,
            self.friction_coef.view(-1, 1), self.restitution_coef.view(-1, 1),
        ), dim=-1).clamp(-self.cfg.normalization.clip_observations,
                         self.cfg.normalization.clip_observations)

    def compute_observations(self):
        if self.odometry is not None:
            fresh = ~self.sensor_ready
            if fresh.any():
                self.sensor_cache[fresh] = self._measure_proprioception()[fresh]
                q, _, gravity = self._odometry_measurements()
                self.odometry.reset(fresh.nonzero(as_tuple=False).flatten(), q, gravity)
                self.sensor_ready[fresh] = True
            self._parking_observation()
            obs = torch.cat((self.sensor_cache, self.odometry_features), dim=-1)
            self.privileged_obs_buf = self._critic_observations(self.compute_proprioception_observations())
        else:
            obs = self.compute_proprioception_observations()
            self.privileged_obs_buf = self._critic_observations(obs)
            if self.add_noise:
                obs = obs + (2*torch.rand_like(obs)-1)*self.noise_scale_vec
        limit = self.cfg.normalization.clip_observations
        self.obs_buf = obs.clamp(-limit, limit)
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], self.obs_buf), dim=-1)
        # The same first noisy frame fills history; no prior-episode samples.
        self.obs_history[self.history_reset] = self.obs_buf[self.history_reset].repeat(1, self.obs_history_length)
        self.history_reset.zero_()

    def check_termination(self):
        super().check_termination()
        physical_failure = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices], dim=-1) > 5.0, dim=1)
        physical_failure |= self.projected_gravity[:, 2] > -math.cos(self.cfg.env.max_tilt_rad)
        physical_failure |= self.base_height < self.cfg.env.min_base_height
        self.physical_failure = physical_failure
        previously_locked = self.parking.locked.clone()
        self.parking.update(self.root_states[:, :3], self.commands, self.dt, self.cfg.rewards)
        if self.odometry is not None:
            self.sensor_cache[:] = self._measure_proprioception()
            self.sensor_ready[:] = True
            q, gyro, gravity = self._odometry_measurements()
            if self.cfg.odometry.observe_wheel_increments:
                delta = q[:, [2, 5]]-self.odometry.previous_q[:, [2, 5]]
                self.wheel_increment[:] = torch.atan2(torch.sin(delta), torch.cos(delta))
                self.wheel_increment[~self.odometry.initialized] = 0
                self.sensor_cache[:, [11, 14]] = self.wheel_increment/self.cfg.odometry.increment_scale
            self.odometry.update(q, gyro, gravity, self.dt)
            lock = self.parking.stationary & self.parking.locked & ~previously_locked
            self.odometry_reference[lock] = self.odometry.xy[lock]
            self._parking_observation()
        cfg = self.cfg.estimator
        force = self.contact_forces[:, self.feet_indices, 2].clamp_min(0)
        self.wheel_contact[:], self.contact_counter[:] = update_contact(
            force, self.wheel_contact, self.contact_counter, cfg.contact_on_force,
            cfg.contact_off_force, cfg.contact_confirm_frames)
        slip = flat_wheel_slip(self.rigid_body_states[:, self.feet_indices],
            self.wheel_center_local, self.wheel_com_local, self.cfg.asset.wheel_radius,
            cfg.wheel_width, self.base_quat)
        # Instantaneous support mask excludes airborne frames during debounce.
        support = force > cfg.contact_off_force
        self.estimator_targets[:] = torch.cat((
            self.base_lin_vel/cfg.velocity_scale, self.wheel_contact.float(),
            slip/cfg.slip_scale, torch.ones(self.num_envs, 1, device=self.device),
            support.float()), dim=-1)
        gravity = self.projected_gravity
        pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=-1))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        # Snapshot before automatic resets overwrite state, contact or position.
        self.transition_metrics = {
            "velocity": self.base_lin_vel.clone(), "position": self.root_states[:, :3].clone(),
            "pitch": pitch.clone(), "roll": roll.clone(), "slip": slip.clone(),
            "support": support.clone(), "contact": self.wheel_contact.clone(),
            "force": force.clone(), "command": self.commands[:, 0].clone(),
            "saturation": (self.torques.abs() >= 0.98*self.torque_limits).float().mean(-1),
            "failure": physical_failure,
            "yaw_rate": self.base_ang_vel[:,2].clone(),
            "yaw_command": self.commands[:,1].clone(),
            "target_command": self.command_targets[:, 0].clone(),
            "command_acceleration": self.command_acceleration.clone(),
            "linear_acceleration": quat_rotate_inverse(
                self.base_quat, (self.root_states[:, 7:10]-self.previous_world_velocity)/self.dt)[:, 0],
            "targets": self.estimator_targets.clone(),
            "parking_valid": (self.parking.stationary & self.parking.locked).clone(),
            "parking_error": self.parking.error.clone(),
            "braking_distance": self.parking.braking_distance.clone(),
            "action_clipping": self.raw_action_clipping.clone(),
        }
        self.previous_world_velocity[:] = self.root_states[:, 7:10]
        self.terminal_critic_obs[:] = self._critic_observations(self.compute_proprioception_observations())

    def reset_idx(self, env_ids):
        if len(env_ids) and self.odometry is not None:
            self.wheel_increment[env_ids] = 0
            self.odometry_features[env_ids] = 0
            self.sensor_ready[env_ids] = False
            self.odometry.initialized[env_ids] = False
            self.odometry_reference[env_ids] = 0
        super().reset_idx(env_ids)
        if len(env_ids):
            if self.cfg.commands.max_linear_acceleration > 0:
                # A new episode starts from a zero command. Resampling may
                # choose a nonzero target, but the target is reached by a ramp.
                self._apply_sequence_commands(env_ids)
                self.command_targets[env_ids] = self.commands[env_ids]
                self.commands[env_ids, 0] = 0.0
            self.linear_command_state[env_ids] = self.commands[env_ids, 0]
            self.command_acceleration[env_ids] = 0.0
            self.previous_world_velocity[env_ids] = self.root_states[env_ids, 7:10]
            self.wheel_contact[env_ids] = False
            self.contact_counter[env_ids] = 0
            self.estimator_targets[env_ids] = 0  # reset state has no refreshed contact label
            self.history_reset[env_ids] = True
            self.pulse_force[env_ids] = 0
            self.eval_force[env_ids] = 0
            self.parking.reset(env_ids, self.root_states[:, :3])
