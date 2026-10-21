"""Joint PD Q1 task; relative references are observable by the policy."""
import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

from wheel_legged_gym.envs.q1_flat.q1_flat import Q1Flat


DIAGNOSTIC_FIELDS = [
    "target_vx", "applied_vx_command", "next_vx_command", "vx", "vy", "yaw_rate",
    "roll", "pitch", "height", "worst_leg_error",
    "wheel_speed_l", "wheel_speed_r", "wheel_torque_l", "wheel_torque_r",
    "wheel_force_l", "wheel_force_r", "wheel_duty_l", "wheel_duty_r",
    "wheel_clearance_l", "wheel_clearance_r", "slip_l", "slip_r",
    "saturation_l", "saturation_r", "parking_drift", "cross_track", "heading_error",
    "parking_weight", "parking_active", "l_thigh", "l_calf", "r_thigh", "r_calf",
    "target_l_thigh", "target_l_calf", "target_r_thigh", "target_r_calf",
    "target_wheel_l", "target_wheel_r",
    "target_yaw_rate", "applied_yaw_command", "next_yaw_command",
]


class Q1(Q1Flat):
    def _init_buffers(self):
        super()._init_buffers()
        self.rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, self.num_bodies, 13)
        self.command_targets = torch.zeros_like(self.commands)
        self.command_targets[:, 2] = self.cfg.commands.ranges.height[0]
        self.external_commands = False
        self.idle_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.parking_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.path_active = torch.zeros_like(self.parking_active)
        self.parking_reference = torch.zeros(self.num_envs, 2, device=self.device)
        self.path_reference = torch.zeros_like(self.parking_reference)
        self.heading_reference = torch.zeros(self.num_envs, device=self.device)
        self.parking_error_body = torch.zeros_like(self.parking_reference)
        self.cross_track = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros_like(self.cross_track)
        self.command_acceleration = torch.zeros_like(self.cross_track)
        self.yaw_acceleration = torch.zeros_like(self.cross_track)
        self.parking_weight = torch.zeros_like(self.cross_track)
        self.wheel_clearance = torch.full((self.num_envs, 2), 0.00534, device=self.device)
        self.wheel_force = torch.zeros_like(self.wheel_clearance)
        self.wheel_duty = torch.zeros_like(self.wheel_clearance)
        self.wheel_slip = torch.zeros_like(self.wheel_clearance)
        self.wheel_saturation = torch.zeros_like(self.wheel_clearance)
        self.metric_sum = torch.zeros(self.num_envs, 7, device=self.device)
        self.metric_steps = torch.zeros(self.num_envs, device=self.device)
        self.applied_commands = self.commands.clone()
        self.position_target = torch.zeros_like(self.dof_pos)
        self.velocity_target = torch.zeros_like(self.dof_vel)
        self.last_diagnostics = torch.zeros(self.num_envs, len(DIAGNOSTIC_FIELDS), device=self.device)
        self.wheel_axis_local = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.wheel_axis_local[:, :, 1] = 1.0
        self.wheel_center_local = torch.zeros_like(self.wheel_axis_local)
        self.wheel_center_local[:, :, 1] = torch.tensor(self.cfg.rewards.wheel_center_y, device=self.device)
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.wheel_com_local = torch.tensor([
            [props[index].com.x, props[index].com.y, props[index].com.z]
            for index in self.feet_indices.cpu().tolist()
        ], device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        self.world_up = torch.zeros_like(self.wheel_axis_local)
        self.world_up[:, :, 2] = 1.0
        self.body_forward = torch.zeros(self.num_envs, 3, device=self.device)
        self.body_forward[:, 0] = 1.0
        self.wheel_lateral_offsets = torch.tensor(self.cfg.asset.wheel_lateral_offsets, device=self.device)

    def _resample_commands(self, env_ids):
        if self.external_commands or len(env_ids) == 0:
            return
        maximum = max(abs(x) for x in self.cfg.commands.ranges.lin_vel_x)
        cfg = self.cfg.commands
        yaw_max = max(abs(x) for x in cfg.ranges.ang_vel_yaw)
        minimum = min(cfg.minimum_motion_speed, maximum)
        magnitude = minimum + torch.rand(len(env_ids), device=self.device) * (maximum-minimum)
        endpoint = torch.rand(len(env_ids), device=self.device) < cfg.endpoint_probability
        magnitude = torch.where(endpoint, torch.full_like(magnitude, maximum), magnitude)
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
        stand = torch.rand(len(env_ids), device=self.device) < cfg.stand_probability
        if maximum == 0 and yaw_max == 0:
            stand[:] = True
        self.command_targets[env_ids, 0] = torch.where(stand, torch.zeros_like(magnitude), magnitude * sign)
        self.command_targets[env_ids, 1] = 0.0
        if yaw_max > 0:
            mode = torch.rand(len(env_ids), device=self.device)
            straight = mode < cfg.straight_probability
            spin = (mode >= cfg.straight_probability) & (mode < cfg.straight_probability+cfg.in_place_probability)
            yaw_min = min(cfg.minimum_turn_rate, yaw_max)
            yaw = yaw_min + torch.rand(len(env_ids), device=self.device)*(yaw_max-yaw_min)
            endpoint = torch.rand(len(env_ids), device=self.device) < cfg.endpoint_probability
            yaw = torch.where(endpoint, torch.full_like(yaw, yaw_max), yaw)
            yaw *= torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
            self.command_targets[env_ids, 1] = torch.where(stand | straight, torch.zeros_like(yaw), yaw)
            self.command_targets[env_ids[spin], 0] = 0.0
        self.command_targets[env_ids, 2] = cfg.ranges.height[0]

    def set_command_target(self, speed, yaw_rate=0.0):
        """Playback targets [vx, yaw rate]; both use the training slew limits."""
        self.external_commands = True
        self.command_targets[:, 0] = speed
        self.command_targets[:, 1] = yaw_rate
        self.command_targets[:, 2] = self.cfg.commands.ranges.height[0]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        delta = (self.command_targets[:, 0] - self.commands[:, 0]).clamp(
            -self.cfg.commands.acceleration_limit * self.dt,
            self.cfg.commands.acceleration_limit * self.dt
        )
        self.commands[:, 0] += delta
        yaw_delta = (self.command_targets[:, 1] - self.commands[:, 1]).clamp(
            -self.cfg.commands.yaw_acceleration_limit*self.dt,
            self.cfg.commands.yaw_acceleration_limit*self.dt
        )
        self.commands[:, 1] += yaw_delta
        self.yaw_acceleration[:] = yaw_delta/self.dt
        self.commands[:, 2] = self.command_targets[:, 2]
        self.command_acceleration[:] = delta / self.dt
        self._update_references()
        self.metric_sum += torch.stack((
            (self.applied_commands[:, 0] - self.base_lin_vel[:, 0]).abs(),
            (self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]).abs().amax(1),
            self.wheel_duty.amin(1), self.wheel_slip.abs().mean(1),
            self.wheel_saturation.mean(1), self.parking_error_body.norm(dim=1),
            (self.applied_commands[:, 1]-self.base_ang_vel[:, 2]).abs(),
        ), dim=1)
        self.metric_steps += 1
        self._record_diagnostics()

    def _record_diagnostics(self):
        q = self.base_quat
        roll = torch.atan2(2*(q[:, 3]*q[:, 0]+q[:, 1]*q[:, 2]), 1-2*(q[:, 0].square()+q[:, 1].square()))
        pitch = torch.asin((2*(q[:, 3]*q[:, 1]-q[:, 2]*q[:, 0])).clamp(-1, 1))
        column = lambda x: x[:, None]
        self.last_diagnostics[:] = torch.cat((
            self.command_targets[:, :1], self.applied_commands[:, :1], self.commands[:, :1],
            self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3], column(roll), column(pitch), column(self.base_height),
            column((self.dof_pos[:, self.leg_dof_indices]-self.default_dof_pos[:, self.leg_dof_indices]).abs().amax(1)),
            self.dof_vel[:, self.wheel_dof_indices], self.torques[:, self.wheel_dof_indices],
            self.wheel_force, self.wheel_duty, self.wheel_clearance, self.wheel_slip, self.wheel_saturation,
            column(self.parking_error_body.norm(dim=1)), column(self.cross_track), column(self.heading_error),
            column(self.parking_weight), column(self.parking_active.float()), self.dof_pos[:, self.leg_dof_indices],
            self.position_target[:, self.leg_dof_indices], self.velocity_target[:, self.wheel_dof_indices],
            self.command_targets[:, 1:2], self.applied_commands[:, 1:2], self.commands[:, 1:2],
        ), dim=1)

    def _update_references(self):
        cfg = self.cfg.commands
        q = self.base_quat
        yaw = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1].square()+q[:, 2].square()))
        target_idle = (self.command_targets[:, 0].abs() < cfg.idle_threshold) & (self.command_targets[:, 1].abs() < cfg.yaw_idle_threshold)
        idle = target_idle & (self.commands[:, 0].abs() < cfg.idle_threshold) & (self.commands[:, 1].abs() < cfg.yaw_idle_threshold)
        self.idle_elapsed[:] = torch.where(idle, self.idle_elapsed + self.dt, torch.zeros_like(self.idle_elapsed))
        parking = idle & (self.idle_elapsed >= cfg.parking_grace_s)
        entering = parking & ~self.parking_active
        self.parking_reference[entering] = self.root_states[entering, :2]
        self.heading_reference[entering] = yaw[entering]
        # A fixed straight-line reference must not oppose requested turns.
        # Include the ramp: re-anchor only after the yaw command settles to zero.
        turning = (self.command_targets[:, 1].abs() >= cfg.yaw_idle_threshold) | (self.commands[:, 1].abs() >= cfg.yaw_idle_threshold)
        moving = (~target_idle) & ~turning
        entering_motion = moving & ~self.path_active
        self.path_reference[entering_motion] = self.root_states[entering_motion, :2]
        self.heading_reference[entering_motion] = yaw[entering_motion]
        self.parking_active[:] = parking
        self.path_active[:] = moving
        self.parking_weight[:] = ((self.idle_elapsed-cfg.parking_grace_s) / cfg.parking_ramp_s).clamp(0, 1) * parking
        delta = self.root_states[:, :2] - self.parking_reference
        self.parking_error_body[:, 0] = (torch.cos(yaw)*delta[:, 0] + torch.sin(yaw)*delta[:, 1]) * parking
        self.parking_error_body[:, 1] = (-torch.sin(yaw)*delta[:, 0] + torch.cos(yaw)*delta[:, 1]) * parking
        delta_path = self.root_states[:, :2] - self.path_reference
        self.cross_track[:] = (-torch.sin(self.heading_reference)*delta_path[:, 0]
                               + torch.cos(self.heading_reference)*delta_path[:, 1]) * moving
        error = yaw-self.heading_reference
        self.heading_error[:] = torch.atan2(torch.sin(error), torch.cos(error)) * (moving | parking)

    def compute_proprioception_observations(self):
        cfg = self.cfg.rewards
        references = torch.cat((self.parking_error_body / cfg.parking_position_scale,
                                self.cross_track[:, None] / cfg.path_position_scale,
                                self.heading_error[:, None] / cfg.heading_scale), dim=1).clamp(-5, 5)
        return torch.cat((super().compute_proprioception_observations(), references), dim=1)

    def compute_observations(self):
        super().compute_observations()
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            self.wheel_clearance * 10.0,
                                            self.wheel_force * 0.02), dim=1)

    def _compute_torques(self, actions):
        position_ref, velocity_ref = torch.zeros_like(self.dof_pos), torch.zeros_like(self.dof_vel)
        position_ref[:, self.leg_dof_indices] = actions[:, [0, 1, 3, 4]] * self.cfg.control.pos_action_scale
        wheel_ref = actions[:, [2, 5]] * self.cfg.control.vel_action_scale
        if self.cfg.control.wheel_velocity_feedforward:
            wheel_ref = wheel_ref + (self.commands[:, 0:1]
                                     - self.commands[:, 1:2]*self.wheel_lateral_offsets) / self.cfg.asset.wheel_radius
        limit = self.dof_vel_limits[self.wheel_dof_indices]
        velocity_ref[:, self.wheel_dof_indices] = torch.clip(wheel_ref, -limit, limit)
        self.position_target[:] = position_ref+self.default_dof_pos
        self.velocity_target[:] = velocity_ref
        raw = (self.p_gains*(position_ref+self.default_dof_pos-self.dof_pos)
               + self.d_gains*(velocity_ref-self.dof_vel))*self.torques_scale
        return torch.clip(raw, -self.torque_limits, self.torque_limits)

    def _sample_wheel_state(self):
        """World angular velocity includes leg and base rotation; no v=r*qdot shortcut."""
        state = self.rigid_body_states[:, self.feet_indices]
        cfg = self.cfg.rewards
        quaternion = state[:, :, 3:7].reshape(-1, 4)
        axis = quat_apply(quaternion, self.wheel_axis_local.reshape(-1, 3)).view_as(self.wheel_axis_local)
        center_offset = quat_apply(quaternion, self.wheel_center_local.reshape(-1, 3)).view_as(self.wheel_center_local)
        com_offset = quat_apply(quaternion, self.wheel_com_local.reshape(-1, 3)).view_as(self.wheel_com_local)
        center = state[:, :, :3] + center_offset
        perpendicular = self.world_up - axis[:, :, 2:3] * axis
        radial_norm = perpendicular.norm(dim=2, keepdim=True).clamp_min(1e-6)
        radius = self.cfg.asset.wheel_radius
        bottom_offset = -radius * perpendicular / radial_norm - 0.5*cfg.wheel_width*axis[:, :, 2:3].sign()*axis
        clearance = center[:, :, 2] + bottom_offset[:, :, 2]
        # PhysX link linear velocity is at COM, while pose is the link frame.
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/_build/physx/latest/class_px_articulation_link.html
        contact_velocity = state[:, :, 7:10] + torch.cross(state[:, :, 10:13], center_offset+bottom_offset-com_offset, dim=-1)
        forward = quat_apply(self.base_quat, self.body_forward)
        forward[:, 2] = 0.0
        forward = forward / forward.norm(dim=1, keepdim=True).clamp_min(1e-6)
        slip = (contact_velocity * forward[:, None]).sum(dim=2)
        force = self.contact_forces[:, self.feet_indices, 2].clamp_min(0)
        touch = (force > cfg.support_force) & (clearance < cfg.clearance_allowance)
        return clearance, force, touch.float(), slip * touch

    def step(self, actions):
        self.actions = actions.to(self.device).clamp(-self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        self.applied_commands[:] = self.commands
        self.render()
        self.pre_physics_step()
        for field in (self.wheel_force, self.wheel_duty, self.wheel_slip, self.wheel_saturation):
            field.zero_()
        self.wheel_clearance[:] = -1.0
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat((self.actions[:, None], self.action_fifo[:, :-1]), dim=1)
            delayed = self.action_fifo[torch.arange(self.num_envs, device=self.device), self.action_delay_idx]
            self.torques = self._compute_torques(delayed)
            self.wheel_saturation += (self.torques[:, self.wheel_dof_indices].abs()
                                      >= 0.99*self.torque_limits[self.wheel_dof_indices]).float()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.base_quat[:] = self.root_states[:, 3:7]
            clearance, force, touch, slip = self._sample_wheel_state()
            self.wheel_clearance[:] = torch.maximum(self.wheel_clearance, clearance)
            self.wheel_force += force
            self.wheel_duty += touch
            self.wheel_slip += slip.abs()
        for field in (self.wheel_force, self.wheel_duty, self.wheel_slip, self.wheel_saturation):
            field /= self.cfg.control.decimation
        self.post_physics_step()
        self.obs_buf = self.obs_buf.clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        self.privileged_obs_buf = self.privileged_obs_buf.clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.obs_history

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        valid = env_ids[self.metric_steps[env_ids] > 0]
        metrics = None if len(valid) == 0 else (self.metric_sum[valid] / self.metric_steps[valid, None]).mean(dim=0)
        for field in (self.command_targets, self.commands, self.idle_elapsed, self.parking_weight,
                      self.parking_error_body, self.cross_track, self.heading_error, self.command_acceleration, self.yaw_acceleration,
                      self.parking_active, self.path_active, self.wheel_force, self.wheel_duty,
                      self.wheel_slip, self.wheel_saturation):
            field[env_ids] = 0
        self.command_targets[env_ids, 2] = self.cfg.commands.ranges.height[0]
        self.commands[env_ids, 2] = self.cfg.commands.ranges.height[0]
        self.wheel_clearance[env_ids] = 0.00534
        super().reset_idx(env_ids)
        if metrics is not None:
            for name, value in zip(("speed_mae", "worst_leg_error", "both_wheel_duty", "slip_speed", "wheel_saturation", "parking_drift", "yaw_mae"), metrics):
                self.extras["episode"]["q1_"+name] = value
        self.metric_sum[env_ids] = 0
        self.metric_steps[env_ids] = 0

    def check_termination(self):
        super().check_termination()
        # Physical failure wins when a fall coincides with the time limit.
        self.physical_failure = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1) > 5.0, dim=1)
        self.physical_failure |= self.projected_gravity[:, 2] > -math.cos(self.cfg.env.max_tilt_rad)
        self.physical_failure |= self.base_height < self.cfg.env.min_base_height
        self.time_out_buf &= ~self.physical_failure
        self.reset_buf = self.physical_failure | self.time_out_buf

    def _quality_weight(self):
        tilt = self.projected_gravity[:, :2].norm(dim=1)
        recovery = (1.0-tilt/0.35).clamp(0.2, 1.0)
        acceleration = (self.command_acceleration.abs()/self.cfg.commands.acceleration_limit).clamp(0, 1)
        acceleration = torch.maximum(acceleration, (self.yaw_acceleration.abs()/self.cfg.commands.yaw_acceleration_limit).clamp(0, 1))
        return recovery*(1.0-0.5*acceleration)

    def _reward_tracking_lin_vel(self):
        sigma = torch.where(self.parking_active,
                            torch.full_like(self.base_height, self.cfg.rewards.idle_tracking_sigma),
                            torch.full_like(self.base_height, self.cfg.rewards.tracking_sigma))
        return torch.exp(-(self.applied_commands[:, 0]-self.base_lin_vel[:, 0]).square()/sigma)

    def _reward_tracking_ang_vel(self):
        return torch.exp(-(self.applied_commands[:, 1]-self.base_ang_vel[:, 2]).square()/self.cfg.rewards.yaw_tracking_sigma)

    def _reward_turning_yaw_tracking(self):
        # Same principle as four-legged pure-turn tracking, with Q1 channel 1.
        pure_turn = (self.applied_commands[:, 0].abs() < self.cfg.commands.idle_threshold)
        pure_turn &= self.applied_commands[:, 1].abs() >= self.cfg.commands.yaw_idle_threshold
        return pure_turn.float()*self._reward_tracking_ang_vel()

    def _reward_speed_error(self):
        return ((self.applied_commands[:, 0]-self.base_lin_vel[:, 0]).abs()-0.02).clamp_min(0).div(0.15).clamp(max=2)

    def _reward_actual_posture(self):
        error = (self.dof_pos[:, self.leg_dof_indices]-self.default_dof_pos[:, self.leg_dof_indices]).abs()
        cost = ((error-self.cfg.rewards.pose_allowance).clamp_min(0)/self.cfg.rewards.pose_error_scale).square().clamp(max=4)
        return (cost.mean(1)+0.5*cost.amax(1))*self._quality_weight()

    def _reward_leg_symmetry(self):
        pose = self.dof_pos[:, self.leg_dof_indices].reshape(self.num_envs, 2, 2)
        turning = (self.commands[:, 1].abs()/0.6).clamp(0, 1)
        return ((pose[:, 0]-pose[:, 1]).abs()-self.cfg.rewards.symmetry_allowance).clamp_min(0).div(0.2).square().mean(1).clamp(max=4)*self._quality_weight()*(1.0-0.5*turning)

    def _reward_rolling_contact(self):
        clearance = ((self.wheel_clearance-self.cfg.rewards.clearance_allowance).clamp_min(0)/self.cfg.rewards.clearance_scale).square().clamp(max=4)
        missing = 1.0-self.wheel_duty
        startup = (self.episode_length_buf*self.dt/0.2).clamp(0, 1)
        return (missing.mean(1)+0.5*missing.amax(1)+clearance.mean(1))*startup

    def _reward_rolling_leg_motion(self):
        return self.dof_vel[:, self.leg_dof_indices].square().mean(1)*self._quality_weight()

    def _reward_rolling_slip(self):
        return ((self.wheel_slip-self.cfg.rewards.slip_deadband).clamp_min(0)/self.cfg.rewards.slip_scale).square().clamp(max=4).mean(1)

    def _reward_parking_drift(self):
        return (self.parking_error_body/self.cfg.rewards.parking_position_scale).square().sum(1).clamp(max=4)*self.parking_weight

    def _reward_parking_wheel_speed(self):
        cost = ((self.dof_vel[:, self.wheel_dof_indices].abs()-self.cfg.rewards.wheel_idle_deadband).clamp_min(0)/self.cfg.rewards.wheel_idle_scale).square().clamp(max=4)
        return cost.mean(1)*self.parking_weight*self._quality_weight()

    def _reward_straight_path(self):
        cfg = self.cfg.rewards
        return ((self.cross_track/cfg.path_position_scale).square().clamp(max=4)
                + (self.heading_error/cfg.heading_scale).square().clamp(max=4))*(self.path_active | self.parking_active)

    def _reward_termination(self):
        # Other terms are rates multiplied by dt; a fall is one event, -2 once.
        return self.physical_failure.float()/self.dt
