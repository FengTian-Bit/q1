"""Joint PD Q1 on mixed terrain, with terrain-relative wheel diagnostics."""
import math

import numpy as np
import torch
from isaacgym import gymtorch, terrain_utils
from isaacgym.torch_utils import quat_apply

from wheel_legged_gym.envs.q1_flat.q1_flat import Q1Flat
from wheel_legged_gym.utils.terrain import Terrain


class Q1Terrain(Terrain):
    """Reuse Gym's T0 terrain primitives with explicit Q1 severity ranges."""

    def __init__(self, cfg, num_robots):
        weights = np.asarray(cfg.terrain_proportions, dtype=float)
        if (len(weights) != len(cfg.terrain_names)
                or len(weights) != len(cfg.terrain_labels)
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
            raise ValueError("Q1 terrain names, labels and normalized proportions must agree")
        if not 0 < cfg.stair_height_range[0] <= cfg.stair_height_range[1] <= 0.20:
            raise ValueError("Q1 stair risers must be in (0, 0.20] metres")
        if cfg.num_rows < 1 or cfg.num_cols < 1:
            raise ValueError("Q1 terrain needs at least one row and column")
        self.terrain_classes = np.full((cfg.num_rows, cfg.num_cols), -1, dtype=np.int64)
        super().__init__(cfg, num_robots)

    def curiculum(self):
        if self.cfg.selected:
            self.selected_terrain()
            return
        # Unlike i / num_rows, this includes the configured maximum: 20 cm.
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = row / max(self.cfg.num_rows - 1, 1)
                choice = (col + 0.5) / self.cfg.num_cols
                self.add_terrain_to_map(self.make_terrain(choice, difficulty), row, col)

    def randomized_terrain(self):
        layout = getattr(self.cfg, "playback_layout", None)
        if layout:
            if len(layout) != self.cfg.num_cols or self.cfg.num_rows != 1:
                raise ValueError("Playback needs one explicit terrain column per robot")
            difficulty = self.cfg.playback_difficulty
            if not 0 <= difficulty <= 1:
                raise ValueError("Playback difficulty must be in [0, 1]")
            for col, name in enumerate(layout):
                index = self.cfg.terrain_names.index(name)
                lower = 0.0 if index == 0 else self.proportions[index - 1]
                if self.cfg.terrain_proportions[index] <= 0:
                    raise ValueError("Playback terrain must have a positive proportion")
                choice = (lower + self.proportions[index]) / 2
                self.add_terrain_to_map(self.make_terrain(choice, difficulty), 0, col)
            return
        # Preserve terrain-type coverage even when level curriculum is disabled.
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                self.add_terrain_to_map(self.make_terrain(
                    (col + 0.5) / self.cfg.num_cols, np.random.uniform()), row, col)

    def selected_terrain(self):
        index = self.cfg.terrain_names.index(self.cfg.selected_type)
        lower = 0.0 if index == 0 else self.proportions[index - 1]
        if self.cfg.terrain_proportions[index] <= 0:
            raise ValueError("Selected Q1 terrain must have a positive proportion")
        choice = (lower + self.proportions[index]) / 2
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                self.add_terrain_to_map(self.make_terrain(
                    choice, self.cfg.selected_difficulty), row, col)

    def make_terrain(self, choice, difficulty):
        cfg = self.cfg
        index = min(int(np.searchsorted(self.proportions, choice, side="right")),
                    len(cfg.terrain_names) - 1)
        name = cfg.terrain_names[index]
        difficulty = float(np.clip(difficulty, 0, 1))
        lerp = lambda values: values[0] + (values[1] - values[0]) * difficulty
        terrain = terrain_utils.SubTerrain(
            "q1_" + name, width=self.length_per_env_pixels,
            length=self.width_per_env_pixels, vertical_scale=cfg.vertical_scale,
            horizontal_scale=cfg.horizontal_scale)
        terrain.terrain_type = cfg.terrain_labels[index]
        if name in ("slope_up", "slope_down", "rough_slope"):
            slope = lerp(cfg.rough_slope_range if name == "rough_slope" else cfg.slope_range)
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=-slope if name == "slope_down" else slope,
                platform_size=cfg.platform_size)
            if name == "rough_slope":
                terrain_utils.random_uniform_terrain(
                    terrain, min_height=cfg.rough_height_range[0],
                    max_height=cfg.rough_height_range[1], step=cfg.rough_height_step,
                    downsampled_scale=cfg.rough_downsampled_scale)
        elif name in ("stairs_up", "stairs_down"):
            height = lerp(cfg.stair_height_range)
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=cfg.stair_width,
                step_height=-height if name == "stairs_down" else height,
                platform_size=cfg.platform_size)
        elif name == "discrete":
            terrain_utils.discrete_obstacles_terrain(
                terrain, lerp(cfg.obstacle_height_range), cfg.obstacle_size_range[0],
                cfg.obstacle_size_range[1], cfg.num_obstacles, platform_size=cfg.platform_size)
        elif name != "flat":
            raise ValueError("Unsupported Q1 terrain: " + name)
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        super().add_terrain_to_map(terrain, row, col)
        self.terrain_classes[row, col] = terrain.terrain_type


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
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ("heightfield", "trimesh"):
            self.terrain = Q1Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        else:
            raise ValueError("Q1 requires plane, heightfield or trimesh terrain")
        self._create_envs()

    def _get_env_origins(self):
        self.cfg.terrain.max_init_terrain_level = min(
            self.cfg.terrain.max_init_terrain_level, self.cfg.terrain.num_rows - 1)
        super()._get_env_origins()
        if self.custom_origins:
            self.terrain_class_map = torch.as_tensor(
                self.terrain.terrain_classes, device=self.device, dtype=torch.long)
            self._refresh_terrain_classes()
            self.terrain_y_max = (self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width
                                  + self.cfg.terrain.border_size)

    def _refresh_terrain_classes(self):
        self.env_class = self.terrain_class_map[self.terrain_levels, self.terrain_types]
        for name, label in (("flat", -1), ("smooth_slope", 0), ("rough_slope", 1),
                            ("stair_down", 2), ("stair_up", 3), ("discrete", 4)):
            setattr(self, name + "_idx", (self.env_class == label).nonzero(as_tuple=False).flatten())
        self.basic_terrain_idx = torch.cat((self.flat_idx, self.smooth_slope_idx,
                                           self.rough_slope_idx, self.stair_down_idx))
        self.advanced_terrain_idx = torch.cat((self.stair_up_idx, self.discrete_idx))

    def _flat_terrain_mask(self):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        return self.env_class == -1

    def _terrain_surface(self, xy):
        """Height and upward normal at arbitrary world XY; cliffs stay discrete.

        Smooth cells use the heightfield's triangle interpolation. Vertical
        faces moved by Gym's mesh conversion are approximate here; actual
        contact force remains authoritative for wheel support on rough tiles.
        """
        if self.cfg.terrain.mesh_type == "plane":
            height = torch.zeros_like(xy[..., 0])
            normal = torch.zeros(*height.shape, 3, device=self.device)
            normal[..., 2] = 1
            return height, normal
        cfg = self.cfg.terrain
        grid = (xy + cfg.border_size) / cfg.horizontal_scale
        x = grid[..., 0].clamp(0, self.height_samples.shape[0] - 1.001)
        y = grid[..., 1].clamp(0, self.height_samples.shape[1] - 1.001)
        ix, iy = x.long(), y.long()
        fx, fy = x - ix, y - iy
        h00 = self.height_samples[ix, iy].float() * cfg.vertical_scale
        h10 = self.height_samples[ix + 1, iy].float() * cfg.vertical_scale
        h01 = self.height_samples[ix, iy + 1].float() * cfg.vertical_scale
        h11 = self.height_samples[ix + 1, iy + 1].float() * cfg.vertical_scale
        lower = fx >= fy
        dx = torch.where(lower, h10 - h00, h11 - h01)
        dy = torch.where(lower, h11 - h10, h01 - h00)
        height = h00 + fx*dx + fy*dy
        spread = torch.stack((h00, h10, h01, h11), dim=-1)
        cliff = (spread.amax(-1) - spread.amin(-1)) > cfg.slope_treshold * cfg.horizontal_scale
        height = torch.where(cliff, h00, height)
        dx = torch.where(cliff, torch.zeros_like(dx), dx) / cfg.horizontal_scale
        dy = torch.where(cliff, torch.zeros_like(dy), dy) / cfg.horizontal_scale
        normal = torch.stack((-dx, -dy, torch.ones_like(height)), dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return height, normal

    def _get_heights(self, env_ids=None):
        # Preserve Q1's 77 probes; handle tensor subsets and empty subsets.
        ids = slice(None) if env_ids is None else env_ids
        q = self.base_quat[ids]
        yaw = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]),
                          1-2*(q[:, 1].square()+q[:, 2].square()))
        local = self.height_points[ids]
        xy = torch.stack((torch.cos(yaw)[:, None]*local[..., 0] - torch.sin(yaw)[:, None]*local[..., 1],
                          torch.sin(yaw)[:, None]*local[..., 0] + torch.cos(yaw)[:, None]*local[..., 1]), dim=-1)
        return self._terrain_surface(xy + self.root_states[ids, None, :2])[0]

    def _height_above_support(self):
        """Use loaded wheel bottoms as the rough-ground height reference.

        A trunk XY crossing a riser does not mean its wheels have climbed it.
        Contact-weighted wheel geometry avoids the heightfield's moved cliff
        vertices. When airborne, use conservative sampled ground; body contact
        and tilt remain independent failure checks.
        """
        root_ground = self._terrain_surface(self.root_states[:, :2])[0]
        if self.cfg.terrain.mesh_type == "plane":
            return self.root_states[:, 2] - root_ground
        state = self.rigid_body_states[:, self.feet_indices]
        q = state[..., 3:7].reshape(-1, 4)
        axis = quat_apply(q, self.wheel_axis_local.reshape(-1, 3)).view_as(self.wheel_axis_local)
        offset = quat_apply(q, self.wheel_center_local.reshape(-1, 3)).view_as(self.wheel_center_local)
        center = state[..., :3] + offset
        extent = (self.cfg.asset.wheel_radius * (1-axis[..., 2].square()).clamp_min(0).sqrt()
                  + 0.5*self.cfg.rewards.wheel_width*axis[..., 2].abs())
        bottom = center[..., 2] - extent
        force = self.contact_forces[:, self.feet_indices, 2].clamp_min(0)
        weights = torch.where(force > self.cfg.rewards.support_force, force, torch.zeros_like(force))
        load = weights.sum(dim=1)
        contact_ground = (bottom*weights).sum(dim=1) / load.clamp_min(1e-6)
        foot_ground = self._terrain_surface(center[..., :2])[0].amin(dim=1)
        airborne_ground = torch.minimum(root_ground, foot_ground)
        ground = torch.where(load > 0, contact_ground, airborne_ground)
        return self.root_states[:, 2] - ground

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # Curriculum can move the reset robot to a different height. Do not
        # expose the previous tile's heights in the reset critic observation.
        if not isinstance(self.measured_heights, torch.Tensor):
            self.measured_heights = self._get_heights()
        else:
            self.measured_heights[env_ids] = self._get_heights(env_ids)
        self.base_height[env_ids] = (self.root_states[env_ids, 2]
                                    - self._terrain_surface(self.root_states[env_ids, :2])[0])

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done or len(env_ids) == 0:
            return
        cfg = self.cfg.terrain
        enough = self.metric_steps[env_ids] * self.dt >= cfg.curriculum_min_duration_s
        requested = self.terrain_command_distance[env_ids]
        travelled = self.terrain_travel_distance[env_ids]
        distance = (self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]).norm(dim=1)
        error = self.metric_sum[env_ids, 0] / self.metric_steps[env_ids].clamp_min(1)
        failed = getattr(self, "physical_failure", torch.zeros_like(self.reset_buf))[env_ids]
        move_up = (enough & ~failed & self.time_out_buf[env_ids]
                   & (distance >= cfg.curriculum_min_distance)
                   & (requested >= cfg.curriculum_min_distance)
                   & (travelled >= 0.65*requested) & (error <= cfg.curriculum_tracking_error))
        move_down = ((failed & (self.metric_steps[env_ids]*self.dt >= 1.0))
                     | (enough & (requested >= cfg.curriculum_min_distance)
                        & (travelled < 0.5*requested))) & ~move_up
        self.success_ids, self.fail_ids = env_ids[move_up], env_ids[move_down]
        self.terrain_levels[env_ids] = (self.terrain_levels[env_ids]
                                        + move_up.long() - move_down.long()).clamp(0, self.max_terrain_level-1)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._refresh_terrain_classes()

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
        self.terrain_command_distance = torch.zeros_like(self.metric_steps)
        self.terrain_travel_distance = torch.zeros_like(self.metric_steps)
        self.command_sample_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.needs_command_sample = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
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
        flat = self._flat_terrain_mask()
        rough_period = max(1, int(self.cfg.commands.rough_resampling_time / self.dt))
        resample = (flat[env_ids] | self.needs_command_sample[env_ids]
                    | (self.episode_length_buf[env_ids]-self.command_sample_step[env_ids] >= rough_period))
        env_ids = env_ids[resample]
        if len(env_ids) == 0:
            return
        self.command_sample_step[env_ids] = self.episode_length_buf[env_ids]
        self.needs_command_sample[env_ids] = False
        maximum = max(abs(x) for x in self.cfg.commands.ranges.lin_vel_x)
        cfg = self.cfg.commands
        yaw_max = max(abs(x) for x in cfg.ranges.ang_vel_yaw)
        minimum = min(cfg.minimum_motion_speed, maximum)
        magnitude = minimum + torch.rand(len(env_ids), device=self.device) * (maximum-minimum)
        endpoint = torch.rand(len(env_ids), device=self.device) < cfg.endpoint_probability
        magnitude = torch.where(endpoint, torch.full_like(magnitude, maximum), magnitude)
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
        stand = torch.rand(len(env_ids), device=self.device) < cfg.stand_probability
        stand &= flat[env_ids]
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
        maximum = self.cfg.commands.max_forward_speed
        self.command_targets[:, 0] = torch.as_tensor(speed, device=self.device).clamp(-maximum, maximum)
        self.command_targets[:, 1] = yaw_rate
        self.command_targets[:, 2] = self.cfg.commands.ranges.height[0]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.base_height[:] = self._height_above_support()
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
        self.terrain_command_distance += self.applied_commands[:, 0].abs() * self.dt
        self.terrain_travel_distance += self.base_lin_vel[:, 0].abs() * self.dt
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
        _, normal = self._terrain_surface(center[..., :2])
        axial = (normal * axis).sum(dim=2, keepdim=True)
        perpendicular = normal - axial * axis
        radial_norm = perpendicular.norm(dim=2, keepdim=True).clamp_min(1e-6)
        radius = self.cfg.asset.wheel_radius
        bottom_offset = -radius * perpendicular / radial_norm - 0.5*cfg.wheel_width*axial.sign()*axis
        bottom = center + bottom_offset
        ground_height, _ = self._terrain_surface(bottom[..., :2])
        clearance = (bottom[..., 2] - ground_height) * normal[..., 2]
        # PhysX link linear velocity is at COM, while pose is the link frame.
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/_build/physx/latest/class_px_articulation_link.html
        contact_velocity = state[:, :, 7:10] + torch.cross(state[:, :, 10:13], center_offset+bottom_offset-com_offset, dim=-1)
        forward = quat_apply(self.base_quat, self.body_forward)
        forward[:, 2] = 0.0
        tangent = forward[:, None] - (forward[:, None]*normal).sum(dim=2, keepdim=True)*normal
        tangent = tangent / tangent.norm(dim=2, keepdim=True).clamp_min(1e-6)
        slip = (contact_velocity * tangent).sum(dim=2)
        force = (self.contact_forces[:, self.feet_indices] * normal).sum(dim=2).clamp_min(0)
        # Mesh cliff locations are quantized; never discard real rough-ground
        # support merely because the heightfield contact estimate is offset.
        touch = ((force > cfg.support_force)
                 & ((clearance < cfg.clearance_allowance) | ~self._flat_terrain_mask()[:, None]))
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
        self.needs_command_sample[env_ids] = True
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
        self.command_sample_step[env_ids] = 0
        if metrics is not None:
            for name, value in zip(("speed_mae", "worst_leg_error", "both_wheel_duty", "slip_speed", "wheel_saturation", "parking_drift", "yaw_mae"), metrics):
                self.extras["episode"]["q1_"+name] = value
        self.metric_sum[env_ids] = 0
        self.metric_steps[env_ids] = 0
        self.terrain_command_distance[env_ids] = 0
        self.terrain_travel_distance[env_ids] = 0

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

    def _reward_base_height(self):
        sigma = torch.where(self._flat_terrain_mask(),
                            torch.full_like(self.base_height, self.cfg.rewards.height_sigma),
                            torch.full_like(self.base_height, self.cfg.rewards.rough_height_sigma))
        return torch.exp(-(self.base_height - self.commands[:, 2]).square() / sigma)

    def _reward_speed_error(self):
        scale = self.cfg.rewards.speed_error_scale
        # Cover the full forward/reverse error range instead of saturating at
        # 0.32 m/s, which hid improvement toward the new 0.8 m/s commands.
        maximum_cost = 2.0*self.cfg.commands.max_forward_speed/scale
        return ((self.applied_commands[:, 0]-self.base_lin_vel[:, 0]).abs()-0.02).clamp_min(0).div(scale).clamp(max=maximum_cost)

    def _reward_actual_posture(self):
        error = (self.dof_pos[:, self.leg_dof_indices]-self.default_dof_pos[:, self.leg_dof_indices]).abs()
        cost = ((error-self.cfg.rewards.pose_allowance).clamp_min(0)/self.cfg.rewards.pose_error_scale).square().clamp(max=4)
        return (cost.mean(1)+0.5*cost.amax(1))*self._quality_weight()*self._flat_terrain_mask()

    def _reward_leg_symmetry(self):
        pose = self.dof_pos[:, self.leg_dof_indices].reshape(self.num_envs, 2, 2)
        turning = (self.commands[:, 1].abs()/0.6).clamp(0, 1)
        terrain_weight = torch.where(self._flat_terrain_mask(), 1.0, self.cfg.rewards.rough_leg_symmetry_weight)
        return ((pose[:, 0]-pose[:, 1]).abs()-self.cfg.rewards.symmetry_allowance).clamp_min(0).div(0.2).square().mean(1).clamp(max=4)*self._quality_weight()*(1.0-0.5*turning)*terrain_weight

    def _reward_nominal_state(self):
        return super()._reward_nominal_state()*self._flat_terrain_mask()

    def _reward_rolling_contact(self):
        clearance = ((self.wheel_clearance-self.cfg.rewards.clearance_allowance).clamp_min(0)/self.cfg.rewards.clearance_scale).square().clamp(max=4)
        missing = 1.0-self.wheel_duty
        startup = (self.episode_length_buf*self.dt/0.2).clamp(0, 1)
        flat_cost = missing.mean(1)+0.5*missing.amax(1)+clearance.mean(1)
        # Steps may briefly unload one wheel. Penalize losing both supports;
        # falls still terminate on body collision, excessive tilt or low height.
        rough_cost = missing.prod(dim=1)
        return torch.where(self._flat_terrain_mask(), flat_cost, rough_cost)*startup

    def _reward_rolling_leg_motion(self):
        return self.dof_vel[:, self.leg_dof_indices].square().mean(1)*self._quality_weight()*self._flat_terrain_mask()

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
