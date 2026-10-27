"""Contact-guided climbing with privileged training instructions only.

The policy retains Q1's 31/155 observation interface. The force-driven teacher
never enters actor observations; guidance can be removed completely at evaluation.
"""

import torch
from isaacgym.torch_utils import quat_apply

from wheel_legged_gym.envs.q1.q1 import Q1
from .guidance import ContactGuidance
from .terrain import CTBCTerrain


class Q1CTBC(Q1):
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ("heightfield", "trimesh"):
            self.terrain = CTBCTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "plane":
            self._create_ground_plane()
        else:
            raise ValueError("Q1 CTBC requires plane, heightfield or trimesh terrain")
        self._create_envs()

    def _get_env_origins(self):
        super()._get_env_origins()
        if self.custom_origins:
            self.ctbc_terrain_maps = {
                name: torch.as_tensor(getattr(self.terrain, name), device=self.device)
                for name in ("riser_heights", "tread_widths", "step_counts",
                             "first_riser_x", "goal_x")}
        self._refresh_climb_targets()

    def _refresh_climb_targets(self):
        if self.custom_origins:
            for name, values in self.ctbc_terrain_maps.items():
                setattr(self, "ctbc_" + name,
                        values[self.terrain_levels, self.terrain_types].clone())
        else:
            self.ctbc_riser_heights = torch.zeros(self.num_envs, device=self.device)
            self.ctbc_tread_widths = torch.full_like(self.ctbc_riser_heights, 0.5)
            self.ctbc_step_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.ctbc_first_riser_x = self.env_origins[:, 0] + 0.7
            self.ctbc_goal_x = self.env_origins[:, 0] + self.cfg.terrain.flat_goal_distance
        if hasattr(self, "guidance"):
            self._update_guide_amplitudes()

    def _update_guide_amplitudes(self):
        """Map training-only stair height to a reachable, nominal IK instruction.

        Preserve the nominal wheel X coordinate. The guide remains a heuristic;
        actual clearance also depends on body motion and actuator tracking.
        """
        cfg = self.cfg.ctbc
        bounds = getattr(cfg, "guide_lift_height_range", None)
        angles = self.cfg.init_state.default_joint_angles
        hip0, knee0 = angles["L_thigh_joint"], angles["L_calf_joint"]
        l1, l2 = self.cfg.asset.l1, self.cfg.asset.l2
        q0 = torch.tensor([hip0, knee0], device=self.device)
        x0 = -l1*torch.sin(q0[0])-l2*torch.sin(q0.sum())
        z0 = -l1*torch.cos(q0[0])-l2*torch.cos(q0.sum())
        if bounds is None:
            amplitudes = torch.tensor([cfg.hip_amplitude, cfg.knee_amplitude], device=self.device)
            peak = q0+amplitudes
            lift = -l1*torch.cos(peak[0])-l2*torch.cos(peak.sum())-z0
            self.ctbc_guide_lift_heights = lift.expand(self.num_envs).clone()
            amplitudes = amplitudes.expand(self.num_envs, -1)
        else:
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("CTBC guide lift range must contain positive ordered heights")
            # The user's validated 8 cm guide / 10 cm riser pair is row zero.
            ratio = float(getattr(cfg, "guide_lift_ratio", 0.8))
            if not 0 < ratio <= 1:
                raise ValueError("CTBC guide lift ratio must be in (0, 1]")
            lift = (self.ctbc_riser_heights.float()*ratio).clamp(bounds[0], bounds[1])
            z = z0+lift
            cosine = (x0.square()+z.square()-l1*l1-l2*l2)/(2*l1*l2)
            if torch.any(cosine.abs() > 1.0) or torch.any(z >= 0):
                raise ValueError("CTBC lift reference is outside the nominal leg workspace")
            knee = -torch.acos(cosine)
            hip = torch.atan2(-x0.expand_as(z), -z)-torch.atan2(l2*torch.sin(knee), l1+l2*torch.cos(knee))
            amplitudes = torch.stack((hip-hip0, knee-knee0), dim=1)
            self.ctbc_guide_lift_heights = lift
            available = self.ctbc_leg_limits.reshape(2, 2).amin(dim=0)
            if torch.any(amplitudes.abs() > available+1e-6):
                raise ValueError("CTBC lift instruction exceeds configured leg offset limits")
            if hasattr(self, "dof_pos_limits"):
                limits = self.dof_pos_limits[self.leg_dof_indices].reshape(2, 2, 2)
                lower = limits[..., 0].amax(dim=0)+0.05
                upper = limits[..., 1].amin(dim=0)-0.05
                peak = amplitudes+q0
                if torch.any(peak < lower-1e-6) or torch.any(peak > upper+1e-6):
                    raise ValueError("CTBC lift instruction exceeds joint target margins")
        self.guidance.set_amplitudes(torch.arange(self.num_envs, device=self.device), amplitudes)

    def _init_buffers(self):
        super()._init_buffers()
        self.guidance = ContactGuidance(self.num_envs, self.device, self.dt, self.cfg.ctbc)
        self.ctbc_training_iteration = 0
        self._guidance_mode = "off" if self.cfg.ctbc.pure_policy else "annealed"
        self.ctbc_leg_scales = torch.tensor(self.cfg.control.leg_action_scales, device=self.device)
        if (self.ctbc_leg_scales.shape != (4,) or not torch.isfinite(self.ctbc_leg_scales).all()
                or torch.any(self.ctbc_leg_scales <= 0)):
            raise ValueError("CTBC requires four positive leg action scales")
        limits = getattr(self.cfg.control, "leg_action_limits", None)
        self.ctbc_leg_limits = (self.ctbc_leg_scales.clone() if limits is None
                                else torch.tensor(limits, device=self.device))
        if (self.ctbc_leg_limits.shape != (4,) or not torch.isfinite(self.ctbc_leg_limits).all()
                or torch.any(self.ctbc_leg_limits <= 0)):
            raise ValueError("CTBC requires four finite positive leg offset limits")
        self.ctbc_action_clip = torch.ones(6, device=self.device)
        self.ctbc_action_clip[[0, 1, 3, 4]] = self.ctbc_leg_limits/self.ctbc_leg_scales
        if self.ctbc_action_clip.max() > self.cfg.normalization.clip_actions+1e-6:
            raise ValueError("Global action clip would truncate CTBC per-joint limits")
        self.ctbc_executed_actions = torch.zeros_like(self.actions)
        self.ctbc_raw_actions = torch.zeros_like(self.actions)
        self.ctbc_reference = torch.zeros(self.num_envs, 4, device=self.device)
        self.ctbc_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ctbc_out_of_bounds = torch.zeros_like(self.ctbc_success)
        self.ctbc_goal_dwell = torch.zeros(self.num_envs, device=self.device)
        self.ctbc_best_x = self.root_states[:, 0].clone()
        self.ctbc_progress = torch.zeros_like(self.ctbc_goal_dwell)
        self.ctbc_attempts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ctbc_wins = torch.zeros_like(self.ctbc_attempts)
        self.ctbc_trigger_count = torch.zeros_like(self.ctbc_goal_dwell)
        self.ctbc_swing_timeouts = torch.zeros_like(self.ctbc_goal_dwell)
        if not hasattr(self, "ctbc_goal_x"):
            self._refresh_climb_targets()
        else:
            self._update_guide_amplitudes()

    def set_training_iteration(self, iteration):
        """Called with cumulative PPO updates, including after checkpoint load."""
        self.ctbc_training_iteration = max(int(iteration), 0)

    def set_guidance_mode(self, mode):
        """off: pure network; on: fixed teacher; annealed: training schedule."""
        if mode not in ("off", "on", "annealed"):
            raise ValueError("Unknown CTBC guidance mode: {}".format(mode))
        self._guidance_mode = mode
        self.cfg.ctbc.pure_policy = mode == "off"
        self.guidance.reset(slice(None))

    @property
    def ctbc_feedforward_weight(self):
        if (self._guidance_mode == "off" or self.cfg.ctbc.pure_policy
                or not self.cfg.ctbc.feedforward_enabled):
            return 0.0
        if self._guidance_mode == "on":
            return float(self.cfg.ctbc.feedforward_initial)
        return self.guidance.feedforward_weight(self.ctbc_training_iteration)

    def _resample_commands(self, env_ids):
        if self.external_commands or len(env_ids) == 0:
            return
        cfg = self.cfg.commands
        low, high = cfg.ranges.lin_vel_x
        self.command_targets[env_ids, 0] = low + (high-low)*torch.rand(len(env_ids), device=self.device)
        self.command_targets[env_ids, 1] = 0.0
        self.command_targets[env_ids, 2] = cfg.ranges.height[0]
        self.needs_command_sample[env_ids] = False
        self.command_sample_step[env_ids] = self.episode_length_buf[env_ids]

    def _wheel_geometry(self):
        state = self.rigid_body_states[:, self.feet_indices]
        quaternion = state[..., 3:7].reshape(-1, 4)
        axis = quat_apply(quaternion, self.wheel_axis_local.reshape(-1, 3)).view_as(self.wheel_axis_local)
        offset = quat_apply(quaternion, self.wheel_center_local.reshape(-1, 3)).view_as(self.wheel_center_local)
        center = state[..., :3] + offset
        extent = (self.cfg.asset.wheel_radius*(1-axis[..., 2].square()).clamp_min(0).sqrt()
                  + 0.5*self.cfg.rewards.wheel_width*axis[..., 2].abs())
        return center, center[..., 2]-extent

    def step(self, actions):
        # Never mutate the PPO sample: its log probability belongs to raw policy
        # actions. Q1.step stores the normalized composite as action history.
        raw = torch.maximum(torch.minimum(actions.to(self.device), self.ctbc_action_clip),
                            -self.ctbc_action_clip)
        self.ctbc_raw_actions[:] = raw
        if self._guidance_mode != "off" and not self.cfg.ctbc.pure_policy:
            _, bottom = self._wheel_geometry()
            force = self.contact_forces[:, self.feet_indices]
            self.guidance.update(force[..., :2].norm(dim=-1),
                                 force[..., 2] > self.cfg.rewards.support_force,
                                 bottom, self.commands[:, 0] > self.cfg.commands.idle_threshold)
        self.ctbc_reference[:] = self.guidance.reference_offsets()
        composite = raw.clone()
        composite[:, [0, 1, 3, 4]] += (self.ctbc_feedforward_weight*self.ctbc_reference
                                       / self.ctbc_leg_scales)
        composite = torch.maximum(torch.minimum(composite, self.ctbc_action_clip),
                                  -self.ctbc_action_clip)
        return super().step(composite)

    def _compute_torques(self, actions):
        position_ref = actions[:, [0, 1, 3, 4]]*self.ctbc_leg_scales
        target = self.default_dof_pos[:, self.leg_dof_indices] + position_ref
        # Keep a physical margin even if future scales/default-pose DR change.
        limits = self.dof_pos_limits[self.leg_dof_indices]
        target = torch.maximum(torch.minimum(target, limits[:, 1]-0.05), limits[:, 0]+0.05)
        wheel_ref = actions[:, [2, 5]]*self.cfg.control.vel_action_scale
        if self.cfg.control.wheel_velocity_feedforward:
            wheel_ref += (self.commands[:, 0:1]-self.commands[:, 1:2]*self.wheel_lateral_offsets)/self.cfg.asset.wheel_radius
        limit = self.dof_vel_limits[self.wheel_dof_indices]
        wheel_ref = torch.maximum(torch.minimum(wheel_ref, limit), -limit)
        self.position_target[:] = self.default_dof_pos
        self.position_target[:, self.leg_dof_indices] = target
        self.velocity_target.zero_()
        self.velocity_target[:, self.wheel_dof_indices] = wheel_ref
        self.ctbc_executed_actions[:, [0, 1, 3, 4]] = (
            target-self.default_dof_pos[:, self.leg_dof_indices])/self.ctbc_leg_scales
        self.ctbc_executed_actions[:, [2, 5]] = wheel_ref/self.cfg.control.vel_action_scale
        raw = (self.p_gains*(self.position_target-self.dof_pos)
               + self.d_gains*(self.velocity_target-self.dof_vel))*self.torques_scale
        return torch.maximum(torch.minimum(raw, self.torque_limits), -self.torque_limits)

    def compute_proprioception_observations(self):
        obs = super().compute_proprioception_observations()
        obs[:, 21:27] = self.ctbc_executed_actions
        return obs

    def post_physics_step(self):
        super().post_physics_step()
        # The parent shifts history after computing rewards/observations. Keep
        # its next-step critic and smoothness history in actuator target units,
        # including wheel feedforward and any applied delay or joint clipping.
        self.last_actions[:, :, 0] = self.ctbc_executed_actions

    def _reward_action_rate(self):
        return (self.ctbc_executed_actions-self.last_actions[:, :, 0]).square().sum(dim=1)

    def _reward_action_smooth(self):
        delta = self.ctbc_executed_actions-2*self.last_actions[:, :, 0]+self.last_actions[:, :, 1]
        return delta[:, [0, 1, 3, 4]].square().sum(dim=1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.ctbc_trigger_count += self.guidance.triggered.sum(dim=1)
        self.ctbc_swing_timeouts += self.guidance.timed_out
        self.ctbc_progress[:] = (self.root_states[:, 0]-self.ctbc_best_x).clamp(0.0, 0.03)/self.dt
        self.ctbc_best_x[:] = torch.maximum(self.ctbc_best_x, self.root_states[:, 0])
        center, bottom = self._wheel_geometry()
        top = self.env_origins[:, 2] + self.ctbc_riser_heights*self.ctbc_step_counts
        support = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.support_force
        reached = ((center[..., 0] >= self.ctbc_goal_x[:, None]).all(dim=1)
                   & ((bottom-top[:, None]).abs() <= 0.025).all(dim=1)
                   & support.all(dim=1))
        self.ctbc_goal_dwell[:] = torch.where(reached, self.ctbc_goal_dwell+self.dt,
                                               torch.zeros_like(self.ctbc_goal_dwell))
        self.ctbc_success[:] = self.ctbc_goal_dwell >= self.cfg.ctbc.goal_dwell_s-1e-6

    def check_termination(self):
        super().check_termination()
        # Do not allow a policy to bypass the staircase through an adjacent tile.
        lateral = (self.root_states[:, 1]-self.env_origins[:, 1]).abs()
        self.ctbc_out_of_bounds[:] = lateral > self.cfg.terrain.terrain_width/2-0.25
        self.ctbc_out_of_bounds |= self.root_states[:, 0] < self.env_origins[:, 0]-0.6
        self.ctbc_success &= ~self.physical_failure & ~self.ctbc_out_of_bounds
        self.time_out_buf &= ~self.ctbc_success & ~self.ctbc_out_of_bounds
        self.reset_buf |= self.ctbc_success | self.ctbc_out_of_bounds

    def compute_reward(self):
        super().compute_reward()
        # Base rewards are clipped rates. Completion is a one-off event, like
        # the existing termination penalty, and must retain its configured size.
        scale = self.reward_scales.get("ctbc_success", 0.0)
        if scale:
            event = self._reward_ctbc_success()*scale
            limit = self.cfg.rewards.clip_single_reward*self.dt
            correction = event-event.clamp(-limit, limit)
            self.rew_buf += correction
            self.episode_sums["ctbc_success"] += correction

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done or len(env_ids) == 0:
            return
        valid = env_ids[self.metric_steps[env_ids] > 0]
        rough = valid[self.ctbc_step_counts[valid] > 0]
        self.ctbc_attempts[rough] += 1
        self.ctbc_wins[rough] += self.ctbc_success[rough]
        ready = self.ctbc_attempts[rough] >= self.cfg.ctbc.curriculum_window
        rate = self.ctbc_wins[rough].float()/self.ctbc_attempts[rough].clamp_min(1)
        up = ready & (rate >= self.cfg.ctbc.curriculum_promote_rate)
        down = ready & (rate < self.cfg.ctbc.curriculum_demote_rate)
        self.success_ids, self.fail_ids = rough[up], rough[down]
        self.terrain_levels[rough] = (self.terrain_levels[rough]+up.long()-down.long()).clamp(0, self.max_terrain_level-1)
        self.ctbc_attempts[rough[ready]] = 0
        self.ctbc_wins[rough[ready]] = 0
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._refresh_terrain_classes()
        self._refresh_climb_targets()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        valid = env_ids[self.metric_steps[env_ids] > 0]
        event = {
            "env_ids": valid.clone(), "success": self.ctbc_success[valid].clone(),
            "physical_failure": getattr(self, "physical_failure", self.reset_buf)[valid].clone(),
            "timeout": self.time_out_buf[valid].clone(),
            "height_m": self.ctbc_riser_heights[valid].clone(),
            "steps": self.ctbc_step_counts[valid].clone(),
            "guide_lift_m": self.ctbc_guide_lift_heights[valid].clone(),
            "feedforward_weight": torch.full((len(valid),), self.ctbc_feedforward_weight, device=self.device),
        }
        triggers = self.ctbc_trigger_count[valid].clone()
        swing_timeouts = self.ctbc_swing_timeouts[valid].clone()
        # Keep success until the parent has advanced the curriculum. Clear only
        # actuator history before its reset observation/history is constructed.
        self.ctbc_executed_actions[env_ids] = 0.0
        self.ctbc_raw_actions[env_ids] = 0.0
        self.guidance.reset(env_ids)
        super().reset_idx(env_ids)
        for field in (self.ctbc_reference, self.ctbc_success, self.ctbc_out_of_bounds,
                      self.ctbc_goal_dwell, self.ctbc_progress, self.ctbc_trigger_count,
                      self.ctbc_swing_timeouts):
            field[env_ids] = 0
        self.ctbc_best_x[env_ids] = self.root_states[env_ids, 0]
        if len(valid):
            self.extras["ctbc_episode"] = event
            episode = self.extras["episode"]
            stairs = event["steps"] > 0
            episode["ctbc_success_rate"] = event["success"].float().mean()
            if stairs.any():
                episode["ctbc_stair_success_rate"] = event["success"][stairs].float().mean()
            episode["ctbc_triggers"] = triggers.mean()
            episode["ctbc_swing_timeouts"] = swing_timeouts.mean()
            episode["ctbc_feedforward_weight"] = self.ctbc_feedforward_weight
            episode["ctbc_guide_lift_m"] = event["guide_lift_m"].mean()

    def _reward_leg_symmetry(self):
        """Release symmetry immediately when either wheel hits an obstacle."""
        force_xy = self.contact_forces[:, self.feet_indices, :2].norm(dim=-1)
        # Match ContactGuidance's horizontal force and strict > threshold,
        # independently of its debounce, swing latch and guidance mode.
        collision = (force_xy > self.cfg.ctbc.contact_threshold).any(dim=1)
        return super()._reward_leg_symmetry() * (~collision)

    def _reward_ctbc_target(self):
        actual = self.dof_pos[:, self.leg_dof_indices]
        target = self.default_dof_pos[:, self.leg_dof_indices]+self.ctbc_reference
        error = (actual-target).reshape(self.num_envs, 2, 2).square().mean(dim=2)
        reward = (torch.exp(-error/0.04)*self.guidance.swing_mask).sum(dim=1)
        if getattr(self.cfg.ctbc, "target_reward_anneals", False):
            # Once guidance is removed, do not pull a successful higher lift
            # back toward the smaller teaching trajectory.
            initial = max(float(self.cfg.ctbc.feedforward_initial), 1e-6)
            reward *= min(self.ctbc_feedforward_weight/initial, 1.0)
        return reward

    def _reward_ctbc_clearance(self):
        _, bottom = self._wheel_geometry()
        reference_q = self.default_dof_pos[:, self.leg_dof_indices]+self.ctbc_reference
        reference_q = reference_q.reshape(self.num_envs, 2, 2)
        default_q = self.default_dof_pos[:, self.leg_dof_indices].reshape(self.num_envs, 2, 2)
        fk = lambda q: -self.cfg.asset.l1*torch.cos(q[..., 0])-self.cfg.asset.l2*torch.cos(q.sum(dim=-1))
        desired = fk(reference_q)-fk(default_q)
        actual = bottom-self.guidance.start_height
        allowance = getattr(self.cfg.ctbc, "clearance_overshoot_allowance", 0.0)
        # In high-stair training, a 16 cm guide must not penalize the extra
        # clearance needed for a 20 cm stair. Excessive overshoot remains costly.
        error = (desired-actual).clamp_min(0)+(actual-desired-allowance).clamp_min(0)
        return (torch.exp(-error.square()/0.0009)*self.guidance.swing_mask).sum(dim=1)

    def _reward_ctbc_contact(self):
        swing = self.guidance.swing_mask
        support = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.support_force
        # At trigger/liftoff, first allow time to unload the selected leg.
        match = (support == ~swing).float().mean(dim=1)
        return match*(self.guidance.active_leg >= 0)*(self.guidance.phase >= 0.15)

    def _reward_ctbc_progress(self):
        return self.ctbc_progress

    def _reward_ctbc_success(self):
        return self.ctbc_success.float()/self.dt

    def _reward_ctbc_same_x(self):
        center, _ = self._wheel_geometry()
        return (center[:, 0, 0]-center[:, 1, 0]).abs()
