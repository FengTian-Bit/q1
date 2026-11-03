"""CTBC's six-action Q1 controller with T0 terrain, commands and curriculum.

The old Q1 class remains an internal locomotion base for Q1CTBC. Registering
this subclass as ``q1`` avoids a circular dependency between those classes.
"""
import torch

from wheel_legged_gym.envs.q1_ctbc.q1_ctbc import Q1CTBC
from .q1 import Q1
from .t0_terrain import Q1T0Terrain
from .t0_curriculum import terrain_levels
from .t0_commands import resample_commands, compute_commands


class Q1T0CTBC(Q1CTBC):
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        kind = self.cfg.terrain.mesh_type
        if kind in ("heightfield", "trimesh"):
            self.cfg.terrain.seed = self.cfg.seed
            self.terrain = Q1T0Terrain(self.cfg.terrain, self.num_envs)
        if kind == "trimesh":
            self._create_trimesh()
        elif kind == "heightfield":
            self._create_heightfield()
        elif kind == "plane":
            self._create_ground_plane()
        else:
            raise ValueError("Q1 T0 requires plane, heightfield or trimesh terrain")
        self._create_envs()

    def _get_env_origins(self):
        Q1._get_env_origins(self)
        if self.custom_origins:
            self.ctbc_terrain_maps = {
                name: torch.as_tensor(getattr(self.terrain, name), device=self.device)
                for name in ("riser_heights", "tread_widths", "step_counts")}
        self._refresh_climb_targets()

    def _refresh_climb_targets(self):
        if self.custom_origins:
            for name, values in self.ctbc_terrain_maps.items():
                setattr(self, "ctbc_" + name,
                        values[self.terrain_levels, self.terrain_types].clone())
        else:
            self.ctbc_riser_heights = torch.zeros(self.num_envs, device=self.device)
            self.ctbc_tread_widths = torch.zeros_like(self.ctbc_riser_heights)
            self.ctbc_step_counts = torch.zeros(self.num_envs, device=self.device,
                                                dtype=torch.long)
        # The parent allocates corridor bookkeeping using these buffers. They
        # carry no terminal meaning in this task; its corridor callbacks are bypassed.
        self.ctbc_goal_x = torch.zeros_like(self.ctbc_riser_heights)
        self.ctbc_first_riser_x = torch.zeros_like(self.ctbc_riser_heights)
        if hasattr(self, "guidance"):
            self._update_guide_amplitudes()

    def _init_buffers(self):
        super()._init_buffers()
        self.command_headings = torch.zeros(self.num_envs, device=self.device)
        self._t0_reset_commands = None

    def _resample_commands(self, env_ids):
        resample_commands(self, env_ids)

    def step(self, actions):
        # Gate the teacher by the assigned terrain tile, including any stale
        # swing state left when the curriculum switches terrain at reset.
        stairs = self.ctbc_step_counts > 0
        self.guidance.reset((~stairs).nonzero(as_tuple=False).flatten())
        raw = torch.maximum(torch.minimum(actions.to(self.device), self.ctbc_action_clip),
                            -self.ctbc_action_clip)
        self.ctbc_raw_actions[:] = raw
        if self._guidance_mode != "off" and not self.cfg.ctbc.pure_policy:
            _, bottom = self._wheel_geometry()
            force = self.contact_forces[:, self.feet_indices]
            moving = ((self.commands[:, 0].abs() > self.cfg.commands.idle_threshold)
                      | (self.commands[:, 1].abs() > self.cfg.commands.yaw_idle_threshold))
            self.guidance.update(force[..., :2].norm(dim=-1),
                                 force[..., 2] > self.cfg.rewards.support_force,
                                 bottom, moving & stairs)
        self.ctbc_reference[:] = self.guidance.reference_offsets()
        composite = raw.clone()
        composite[:, [0, 1, 3, 4]] += (self.ctbc_feedforward_weight*self.ctbc_reference
                                      / self.ctbc_leg_scales)
        composite = torch.maximum(torch.minimum(composite, self.ctbc_action_clip),
                                  -self.ctbc_action_clip)
        return Q1.step(self, composite)

    def _post_physics_step_callback(self):
        # T0's heading gain, yaw limit, sampling and lack of command slew differ
        # from the old Q1 callback. Do not call the base's four-channel heading code.
        compute_commands(self)
        self.command_acceleration[:] = (self.commands[:, 0]-self.applied_commands[:, 0])/self.dt
        self.yaw_acceleration[:] = (self.commands[:, 1]-self.applied_commands[:, 1])/self.dt
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height[:] = self._height_above_support()
        self._update_references()
        self.metric_sum += torch.stack((
            (self.applied_commands[:, 0]-self.base_lin_vel[:, 0]).abs(),
            (self.dof_pos[:, self.leg_dof_indices]-self.default_dof_pos[:, self.leg_dof_indices]).abs().amax(1),
            self.wheel_duty.amin(1), self.wheel_slip.abs().mean(1),
            self.wheel_saturation.mean(1), self.parking_error_body.norm(dim=1),
            (self.applied_commands[:, 1]-self.base_ang_vel[:, 2]).abs(),
        ), dim=1)
        self.metric_steps += 1
        self.terrain_command_distance += self.applied_commands[:, 0].abs()*self.dt
        self.terrain_travel_distance += self.base_lin_vel[:, 0].abs()*self.dt
        self._record_diagnostics()
        self.ctbc_trigger_count += self.guidance.triggered.sum(dim=1)
        self.ctbc_swing_timeouts += self.guidance.timed_out

    def check_termination(self):
        # Retain Q1 physical failure and timeout tests, without CTBC's fixed
        # goal, +X-only progress, lateral corridor or backwards escape limits.
        Q1.check_termination(self)

    def _update_terrain_curriculum(self, env_ids):
        # Q1.reset_idx clears commands before the base invokes this hook. T0's
        # downgrade must use the completed episode's command, not that zero.
        saved = self._t0_reset_commands
        if saved is None:
            terrain_levels(self, env_ids)
        else:
            current = self.commands
            try:
                self.commands = saved
                terrain_levels(self, env_ids)
            finally:
                self.commands = current

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        valid = env_ids[self.metric_steps[env_ids] > 0]
        event = {
            "env_ids": valid.clone(),
            "physical_failure": getattr(self, "physical_failure", self.reset_buf)[valid].clone(),
            "timeout": self.time_out_buf[valid].clone(),
            "distance_m": (self.root_states[valid, :2]-self.env_origins[valid, :2]).norm(dim=1),
            "terrain_level": self.terrain_levels[valid].clone() if self.custom_origins
                             else torch.zeros(len(valid), device=self.device, dtype=torch.long),
            "height_m": self.ctbc_riser_heights[valid].clone(),
            "guide_lift_m": self.ctbc_guide_lift_heights[valid].clone(),
        }
        triggers = self.ctbc_trigger_count[valid].clone()
        swing_timeouts = self.ctbc_swing_timeouts[valid].clone()
        self.ctbc_executed_actions[env_ids] = 0
        self.ctbc_raw_actions[env_ids] = 0
        self.guidance.reset(env_ids)
        self._t0_reset_commands = self.commands.clone()
        try:
            Q1.reset_idx(self, env_ids)
        finally:
            self._t0_reset_commands = None
        for field in (self.ctbc_reference, self.ctbc_success, self.ctbc_out_of_bounds,
                      self.ctbc_goal_dwell, self.ctbc_progress, self.ctbc_trigger_count,
                      self.ctbc_swing_timeouts):
            field[env_ids] = 0
        self.ctbc_best_x[env_ids] = self.root_states[env_ids, 0]
        if len(valid):
            self.extras["q1_t0_episode"] = event
            episode = self.extras["episode"]
            episode["q1_t0_distance_m"] = event["distance_m"].mean()
            episode["ctbc_triggers"] = triggers.mean()
            episode["ctbc_swing_timeouts"] = swing_timeouts.mean()
            episode["ctbc_feedforward_weight"] = self.ctbc_feedforward_weight
            episode["ctbc_guide_lift_m"] = event["guide_lift_m"].mean()
