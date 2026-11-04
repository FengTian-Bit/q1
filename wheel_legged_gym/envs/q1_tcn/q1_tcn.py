"""Flat Q1 with aligned state labels and finite-duration trunk force pulses."""

import math
import torch
from isaacgym import gymapi, gymtorch

from wheel_legged_gym.envs.q1_flat.q1_flat import Q1Flat
from .labels import flat_wheel_slip, update_contact
from .parking import ParkingState


class Q1TCN(Q1Flat):
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
        self.terminal_critic_obs = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.parking = ParkingState(self.num_envs, self.device)
        self.raw_action_clipping = torch.zeros(self.num_envs, device=self.device)
        self.reward_terms = {}

    def step(self, actions):
        self.raw_action_clipping = (actions.to(self.device).abs() > self.cfg.normalization.clip_actions).float().mean(-1)
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
        error = (self.commands[:, 1]-self.base_ang_vel[:, 2]).square()
        cfg = self.cfg.rewards
        sigma = cfg.tracking_sigma if cfg.reward_version == 1 else cfg.yaw_tracking_sigma
        return torch.exp(-error/sigma)

    def _reward_parking_drift(self):
        return self.parking.penalty(self.cfg.rewards)

    def _reward_termination(self):
        # Simultaneous timeout must not hide an actual fall in reward v2.
        if self.cfg.rewards.reward_version >= 2:
            return self.physical_failure.float()
        return super()._reward_termination()

    def _resample_commands(self, env_ids):
        if not getattr(self, "external_commands", False):
            super()._resample_commands(env_ids)

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
        self.parking.update(self.root_states[:, :3], self.commands, self.dt, self.cfg.rewards)
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
            "targets": self.estimator_targets.clone(),
            "parking_valid": (self.parking.stationary & self.parking.locked).clone(),
            "parking_error": self.parking.error.clone(),
            "braking_distance": self.parking.braking_distance.clone(),
            "action_clipping": self.raw_action_clipping.clone(),
        }
        self.terminal_critic_obs[:] = self._critic_observations(self.compute_proprioception_observations())

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids):
            self.wheel_contact[env_ids] = False
            self.contact_counter[env_ids] = 0
            self.estimator_targets[env_ids] = 0  # reset state has no refreshed contact label
            self.history_reset[env_ids] = True
            self.pulse_force[env_ids] = 0
            self.eval_force[env_ids] = 0
            self.parking.reset(env_ids, self.root_states[:, :3])
