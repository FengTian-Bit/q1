"""Flat Q1 with aligned state labels and finite-duration trunk force pulses."""

import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
from isaacgym.torch_utils import quat_apply

from wheel_legged_gym.envs.q1_flat.q1_flat import Q1Flat
from .labels import flat_wheel_slip, update_contact
from .parking import ParkingState
from .odometry import WheelOdometry
from .guidance import ContactGuidance
from .q1_tcn_config import configure_observation_layout

# Stage 1 terrain labels live in estimator_targets[:, 10:14]. Scales keep the
# supervised values near unit size: slope normal XY (sin of the tilt, at most
# about 0.26 at 15 deg) x4, wheel clearance in metres x10 (decimetres).
SLOPE_LABEL_SCALE = 4.0
CLEARANCE_LABEL_SCALE = 10.0
CLEARANCE_LABEL_RANGE = (-0.05, 0.30)
# Step labels live in estimator_targets[:, 14:18]: terrain rise (m, x10) at STEP_LOOKAHEAD ahead of each wheel.
STEP_LOOKAHEAD = (0.10, 0.20)
STEP_LABEL_RANGE = (-0.25, 0.30)


def flat_sequence_velocity(time_s, magnitude, sign):
    """Stop/forward/stop/reverse/stop; no phase or clock is an Actor input."""
    phase = time_s.remainder(12.)
    forward = ((phase >= 2.) & (phase < 5.)).to(time_s.dtype)
    reverse = ((phase >= 7.) & (phase < 10.)).to(time_s.dtype)
    return (forward-reverse)*magnitude*sign


def parking_sequence_velocity(time_s, magnitude, sign):
    """Four-second drives separated by six-second parking intervals."""
    phase=time_s.remainder(24.)
    forward=((phase>=4.) & (phase<8.)).to(time_s.dtype)
    reverse=((phase>=14.) & (phase<18.)).to(time_s.dtype)
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


def positive_tracking_reward(error, fine_sigma, coarse_sigma, fine_weight):
    """Broad learning feedback plus a precise peak; sigmas are squared widths."""
    squared_error = error.square()
    coarse = torch.exp(-squared_error/coarse_sigma)
    fine = torch.exp(-squared_error/fine_sigma)
    return (1-fine_weight)*coarse + fine_weight*fine


def command_tracking_sigma(command, idle_sigma, start_speed, full_speed, max_sigma):
    """Squared reward width; smoothstep uses commanded speed, symmetric in vx."""
    fraction = ((command.abs()-start_speed)/(full_speed-start_speed)).clamp(0, 1)
    blend = fraction.square()*(3-2*fraction)
    return idle_sigma+(max_sigma-idle_sigma)*blend


def positive_stand_reward(error, width):
    """One at zero error; a broad tail preserves feedback away from rest."""
    return 1/(1+(error/width).square().mean(dim=-1))


def rate_limit_linear_command(previous, target, dt, acceleration):
    """Limit desired vx changes; never use a sensed or estimated body velocity."""
    if acceleration <= 0:
        return target.clone()
    increment = acceleration*dt
    return previous+(target-previous).clamp(-increment, increment)


def apply_axis_sampling(commands, env_ids, selection, stand_probability, proportions, vx_floor, yaw_floor):
    """Reuse the existing stand draw; only designated parking targets may be zero."""
    choice = (selection-stand_probability)/(1-stand_probability)
    eligible = selection >= stand_probability
    straight = eligible & (choice < proportions['straight'])
    pure_yaw = eligible & (choice >= proportions['straight']) & (
        choice < proportions['straight']+proportions['pure_yaw'])
    # All non-parking components avoid the existing parking dead zone.
    for axis, floor in [(0, vx_floor), (1, yaw_floor)]:
        values = commands[env_ids[eligible], axis]
        sign = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
        commands[env_ids[eligible], axis] = sign*values.abs().clamp_min(floor)
    commands[env_ids[straight], 1] = 0.
    commands[env_ids[pure_yaw], 0] = 0.
    # Optional: redraw part of the combined commands from the high-speed, high-yaw corner.
    high = proportions.get('combined_high', 0.)
    if high > 0:
        combined = eligible & ~straight & ~pure_yaw
        corner = combined & (torch.rand_like(selection) < high)
        ids = env_ids[corner]
        if len(ids):
            band = proportions.get('combined_high_band', [[0.9, 1.2], [0.7, 1.0]])
            for axis, (low, top) in enumerate(band):
                sign = torch.where(commands[ids, axis] >= 0, 1., -1.)
                commands[ids, axis] = sign*(low+(top-low)*torch.rand(len(ids), device=commands.device))


class Q1TCN(Q1Flat):
    def _parse_cfg(self, cfg):
        configure_observation_layout(cfg)
        super()._parse_cfg(cfg)

    def create_sim(self):
        """Plane keeps the inherited path; slopes use the T0 tile generator."""
        if self.cfg.terrain.mesh_type not in ("heightfield", "trimesh"):
            return super().create_sim()
        from wheel_legged_gym.envs.q1.t0_terrain import Q1T0Terrain
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self.cfg.terrain.seed = self.cfg.seed
        self.terrain = Q1T0Terrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "trimesh":
            self._create_trimesh()
        else:
            self._create_heightfield()
        self._create_envs()

    def _terrain_height_at(self, xy):
        """Bilinear terrain height below world XY points. The inherited _get_heights
        takes the minimum of three 10 cm cells, which under-reads a 15 deg slope by
        up to 2.7 cm; that is too coarse for a centimetre clearance label."""
        if self.cfg.terrain.mesh_type == "plane":
            return xy.new_zeros(xy.shape[:-1])
        grid = (xy+self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale
        x0 = grid[..., 0].floor().clamp(0, self.height_samples.shape[0]-2)
        y0 = grid[..., 1].floor().clamp(0, self.height_samples.shape[1]-2)
        fx = (grid[..., 0]-x0).clamp(0, 1)
        fy = (grid[..., 1]-y0).clamp(0, 1)
        px, py = x0.long(), y0.long()
        # Terrain is immutable after creation. Avoid converting the entire
        # height map on every wheel/step label query; rebuild if replaced.
        if getattr(self, "_height_cache_source", None) is not self.height_samples:
            self._height_cache_source = self.height_samples
            self._height_cache_float = self.height_samples.float()
        samples = self._height_cache_float
        heights = ((1-fx)*(1-fy)*samples[px, py]+fx*(1-fy)*samples[px+1, py]
                   + (1-fx)*fy*samples[px, py+1]+fx*fy*samples[px+1, py+1])
        return heights*self.terrain.cfg.vertical_scale

    def _terrain_labels(self):
        """Ground normal XY in the heading frame and per-wheel bottom clearance (metres)."""
        # Least-squares plane z = a*x + b*y + c over the symmetric heading-frame grid.
        coeffs = self.measured_heights @ self.terrain_plane_fit.T
        normal = torch.stack((-coeffs[:, 0], -coeffs[:, 1], torch.ones_like(coeffs[:, 0])), dim=-1)
        slope = (normal/normal.norm(dim=-1, keepdim=True))[:, :2]
        centers = self.rigid_body_states[:, self.feet_indices, :3]
        ground = self._terrain_height_at(centers[..., :2])
        clearance = (centers[..., 2]-self.cfg.asset.wheel_radius-ground).clamp(*CLEARANCE_LABEL_RANGE)
        return slope, clearance

    def _step_labels(self):
        """Terrain rise ahead of each wheel along the heading, ordered [L10, R10, L20, R20]."""
        centers = self.rigid_body_states[:, self.feet_indices, :3]
        forward = quat_apply(self.base_quat, torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 3))[:, :2]
        forward = forward/forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        here = self._terrain_height_at(centers[..., :2])
        rises = [self._terrain_height_at(centers[..., :2]+distance*forward[:, None, :])-here
                 for distance in STEP_LOOKAHEAD]
        return torch.cat(rises, dim=-1).clamp(*STEP_LABEL_RANGE)

    def _get_env_origins(self):
        """Distribute the permanent standing prefix across terrain columns."""
        super()._get_env_origins()
        if not self.custom_origins or not self.cfg.terrain.interleave_command_terrain:
            return
        original = self.terrain_types.clone()
        counts = torch.bincount(original, minlength=self.cfg.terrain.num_cols)
        starts = counts.cumsum(0)-counts
        rank = torch.arange(int(counts.max()), device=self.device)[:, None]
        order = (rank+starts[None, :])[rank < counts[None, :]]
        # The base allocation is column-contiguous. This exact permutation keeps
        # every column's population and consumes no additional random numbers.
        self.terrain_types = original[order]
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(self.num_envs, device=self.device)
        for name in ('flat_idx', 'smooth_slope_idx', 'rough_slope_idx',
                     'stair_up_idx', 'stair_down_idx', 'discrete_idx',
                     'basic_terrain_idx', 'advanced_terrain_idx'):
            setattr(self, name, inverse[getattr(self, name)].sort().values)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _apply_climb_replay(self, env_ids):
        """Reserve one quarter of each up-climb column for a configured riser range.

        Deterministic membership and a per-env cyclic level schedule consume no
        RNG. Other environments retain the existing curriculum. This is a
        training distribution experiment, never an evaluation override.
        """
        if (not getattr(self.cfg.terrain, 'climb_replay_quarter', False)
                or not getattr(self, 'custom_origins', False)
                or not self.cfg.terrain.curriculum
                or self.cfg.terrain.selected
                or getattr(self, 'evaluation_mode', False)
                or not len(env_ids)):
            return
        if not hasattr(self, 'climb_replay_mask'):
            ids = torch.arange(self.num_envs, device=self.device)
            stand_count = int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
            eligible = ids[self._climb_tile_mask(ids) & (ids >= stand_count)]
            self.climb_replay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.climb_replay_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            risers = torch.as_tensor(self.terrain.riser_heights, device=self.device)
            maximum = float(getattr(self.cfg.terrain, 'climb_replay_max_riser', .165))
            if not (.075 <= maximum <= .22):
                raise ValueError('Replay maximum must lie in [0.075,0.22] m')
            self.climb_replay_levels = None
            for col in self.terrain_types[eligible].unique():
                candidates = eligible[self.terrain_types[eligible] == col]
                selected = candidates[::4][:len(candidates)//4]
                levels = torch.nonzero((risers[:, col] >= .075-1e-6)
                                       & (risers[:, col] <= maximum+1e-6)).flatten()
                if not len(levels):
                    raise ValueError('No actual risers in the configured replay range')
                if self.climb_replay_levels is not None and not torch.equal(levels, self.climb_replay_levels):
                    raise ValueError('Replay requires equal height schedules across up-climb columns')
                self.climb_replay_levels = levels
                self.climb_replay_mask[selected] = True
                self.climb_replay_phase[selected] = torch.arange(len(selected), device=self.device) % len(levels)
            self.climb_replay_resets = 0
        selected = env_ids[self.climb_replay_mask[env_ids]]
        if len(selected):
            phase = self.climb_replay_phase[selected]
            self.terrain_levels[selected] = self.climb_replay_levels[phase % len(self.climb_replay_levels)]
            self.climb_replay_phase[selected] += 1
            self.climb_replay_resets += len(selected)

    def _update_terrain_curriculum(self, env_ids):
        """Survival curriculum: v46 commands mix stand, pure yaw and reversals, so the
        base rule (walk 4 m from the origin) would rarely trigger. Promote envs that
        survive and either moved at least curriculum_move_distance or are standing envs;
        demote envs that fell."""
        if not self.init_done or not len(env_ids):
            return
        stand_count = int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
        survived = ~self.physical_failure[env_ids]
        moved = self.max_origin_distance[env_ids] > self.cfg.terrain.curriculum_move_distance
        move_up = survived & (moved | (env_ids < stand_count))
        move_down = ~survived
        if getattr(self.cfg.terrain, 'climb_curriculum', False):
            climb = self._climb_tile_mask(env_ids) & (env_ids >= stand_count)
            riser = self._tile_values('riser_heights', env_ids)
            need = self.cfg.terrain.climb_success_risers*riser-1e-4
            success = survived & (self.climb_max_rise[env_ids] >= need)
            move_up = torch.where(climb, success, move_up)
            move_down = torch.where(climb, ~success, move_down)
            self.climb_episodes += int(climb.sum())
            self.climb_successes += int((climb & success).sum())
            self.climb_falls += int((climb & ~survived).sum())
            if hasattr(self, 'climb_replay_mask'):
                if not hasattr(self, 'climb_group_outcomes'):
                    self.climb_group_outcomes = {g: [0, 0, 0] for g in ('replay', 'progression')}
                for group, membership in [('replay', self.climb_replay_mask[env_ids]),
                                          ('progression', ~self.climb_replay_mask[env_ids])]:
                    selected = climb & membership
                    values = self.climb_group_outcomes[group]
                    values[0] += int(selected.sum())
                    values[1] += int((selected & success).sum())
                    values[2] += int((selected & ~survived).sum())
        self.terrain_levels[env_ids] += move_up.long()-move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            self.terrain_levels[env_ids].clamp_min(0))
        self._apply_climb_replay(env_ids)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._scatter_on_slope(env_ids)

    def _tile_values(self, name, env_ids):
        """Per-env value of a Q1T0Terrain [rows, cols] table for the current tile."""
        cache = getattr(self, '_tile_table_cache', None)
        if cache is None:
            cache = self._tile_table_cache = {}
        if name not in cache:
            table = getattr(getattr(self, 'terrain', None), name, None)
            cache[name] = None if table is None else torch.as_tensor(
                table, device=self.device, dtype=torch.float)
        table = cache[name]
        if table is None:
            return torch.zeros(len(env_ids), device=self.device)
        rows = self.terrain_levels[env_ids].clamp(0, table.shape[0]-1)
        cols = self.terrain_types[env_ids].clamp(0, table.shape[1]-1)
        return table[rows, cols]

    def _climb_tile_mask(self, env_ids):
        """Up-climb tiles: inverted pyramid stairs (class 2), robot drives up from the centre."""
        if not getattr(self, 'custom_origins', False):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        classes = self._tile_values('terrain_classes', env_ids)
        risers = self._tile_values('riser_heights', env_ids)
        return (classes == 2) & (risers > 1e-6)

    def _scatter_on_slope(self, env_ids):
        """Move a fraction of resets from the flat tile centre onto the slope face, so
        holding still on an incline is trained, not only driving onto it."""
        fraction = self.cfg.terrain.slope_spawn_fraction
        if fraction <= 0 or not len(env_ids):
            return
        # Only slope classes are eligible; flat/stair curriculum origins stay fixed.
        classes = self._tile_values('terrain_classes', env_ids)
        eligible = env_ids[(classes == 0) | (classes == 1)]
        chosen = eligible[torch.rand(len(eligible), device=self.device) < fraction]
        if not len(chosen):
            return
        low, high = self.cfg.terrain.slope_spawn_radius
        # A square centre platform extends farther along a diagonal. Keep all
        # sampled angles outside it, with one height-grid cell of margin.
        margin = max(float(self.cfg.terrain.horizontal_scale), 0.1)
        low = max(low, math.sqrt(2)*(self.cfg.terrain.platform_size/2 + margin))
        high = min(high, min(self.cfg.terrain.terrain_length,
                             self.cfg.terrain.terrain_width)/2 - margin)
        if high <= low:
            raise ValueError('Slope spawn radius cannot clear platform within tile')
        radius = low+(high-low)*torch.rand(len(chosen), device=self.device)
        angle = 2*math.pi*torch.rand(len(chosen), device=self.device)
        xy = self.env_origins[chosen, :2]+torch.stack((radius*angle.cos(), radius*angle.sin()), dim=-1)
        self.env_origins[chosen, :2] = xy
        self.env_origins[chosen, 2] = self._terrain_height_at(xy)

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
        self.estimator_targets = torch.zeros(self.num_envs, 18, device=self.device)
        # Pseudo-inverse of [x, y, 1] over the fixed heading-frame height grid.
        grid = self.height_points[0, :, :2]
        design = torch.cat((grid, torch.ones_like(grid[:, :1])), dim=-1)
        self.terrain_plane_fit = torch.linalg.pinv(design)
        self.max_origin_distance = torch.zeros(self.num_envs, device=self.device)
        # stair_rise potential: ground height under the base minus the spawn ground.
        self.stair_spawn_ground = torch.zeros(self.num_envs, device=self.device)
        self.stair_rise_prev = torch.zeros(self.num_envs, device=self.device)
        self.stair_tile = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # r9 climb curriculum: highest world ground rise under the base this episode.
        self.climb_max_rise = torch.zeros(self.num_envs, device=self.device)
        self.climb_episodes = 0
        self.climb_successes = 0
        self.climb_falls = 0
        # The first explicit reset runs the terrain curriculum before any termination check.
        self.physical_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        self._init_ctbc_guidance()

    def _init_ctbc_guidance(self):
        """Stage 2 swing teacher. Always built so reset/step stay branch-free."""
        cfg = self.cfg.ctbc
        self.guidance = ContactGuidance(self.num_envs, self.device, self.dt, cfg)
        self.ctbc_training_iteration = int(cfg.feedforward_start_iteration)
        self.ctbc_reference = torch.zeros(self.num_envs, 4, device=self.device)
        scales = getattr(self.cfg.control, 'leg_action_scales', None)
        self.ctbc_leg_scales = torch.tensor(
            list(scales) if scales is not None else [self.cfg.control.pos_action_scale]*4,
            device=self.device)
        self.ctbc_riser_heights = torch.zeros(self.num_envs, device=self.device)
        self.ctbc_on_stairs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ctbc_lift_table = self._build_guide_lift_table()
        self._refresh_guide_amplitudes(torch.arange(self.num_envs, device=self.device))

    def _build_guide_lift_table(self):
        """Lift (m) -> (thigh, calf) joint offsets (rad) inside this env's ranges.

        For each lift the calf is solved from the vertical reach with the thigh swept
        over its allowed range; the solution with the least horizontal wheel travel
        wins. An airborne wheel may drift, so X is not constrained. An unreachable
        lift raises instead of silently clipping to a smaller trajectory.
        """
        cfg = self.cfg.ctbc
        low, high = cfg.guide_lift_range
        angles = self.cfg.init_state.default_joint_angles
        hip0, knee0 = angles['L_thigh_joint'], angles['L_calf_joint']
        l1, l2 = self.cfg.asset.l1, self.cfg.asset.l2
        limits = getattr(self.cfg.control, 'leg_action_limits', None) or [1.0]*4
        thigh_range = self.ctbc_leg_scales[0].item()*limits[0]
        calf_range = self.ctbc_leg_scales[1].item()*limits[1]
        # leg_dof_indices is [L thigh, L calf, R thigh, R calf]. Thigh (+-1.57) and
        # calf ([-2.6, 0]) limits differ, so the calf rows are selected explicitly.
        urdf = self.dof_pos_limits[self.leg_dof_indices]
        calf_limits = urdf[[1, 3]]
        lower = calf_limits[:, 0].max().item()+self.cfg.control.leg_target_margin
        upper = calf_limits[:, 1].min().item()-self.cfg.control.leg_target_margin
        z0 = -l1*math.cos(hip0)-l2*math.cos(hip0+knee0)
        steps = 161
        lifts = torch.linspace(low, high, steps, device=self.device)
        table = torch.zeros(steps, 2, device=self.device)
        offsets = torch.linspace(-thigh_range, thigh_range, 161, device=self.device)
        for index, lift in enumerate(lifts.tolist()):
            target_z = z0+lift
            hip = hip0+offsets
            cosine = (-target_z-l1*torch.cos(hip))/l2
            valid = cosine.abs() <= 1.0
            knee = -torch.acos(cosine.clamp(-1.0, 1.0))-hip
            valid = valid & ((knee-knee0).abs() <= calf_range+1e-6)
            valid = valid & (knee >= lower-1e-6) & (knee <= upper+1e-6)
            if not bool(valid.any()):
                raise ValueError('CTBC guide lift {:.3f} m is outside the leg workspace'.format(lift))
            x = -l1*torch.sin(hip)-l2*torch.sin(hip+knee)
            cost = x.square().masked_fill(~valid, float('inf'))
            best = int(cost.argmin())
            table[index, 0] = offsets[best]
            table[index, 1] = knee[best]-knee0
        return table

    def _refresh_guide_amplitudes(self, env_ids):
        """Map each robot's current tile riser to its teaching swing amplitude."""
        if not len(env_ids) or not hasattr(self, 'ctbc_lift_table'):
            return
        terrain = getattr(self, 'terrain', None)
        if terrain is not None and hasattr(terrain, 'riser_heights'):
            rows = self.terrain_levels[env_ids].cpu().numpy()
            cols = self.terrain_types[env_ids].cpu().numpy()
            heights = torch.as_tensor(terrain.riser_heights[rows, cols],
                                      device=self.device, dtype=self.ctbc_lift_table.dtype)
        else:
            heights = torch.zeros(len(env_ids), device=self.device,
                                  dtype=self.ctbc_lift_table.dtype)
        self.ctbc_riser_heights[env_ids] = heights
        self.ctbc_on_stairs[env_ids] = False
        # Flat and slope tiles have no riser. Clamping their lift up to the teaching
        # minimum would teach an 8 cm leg lift on flat ground, so they get no lift.
        stairs = heights > 1e-6
        self.ctbc_on_stairs[env_ids] = stairs
        low, high = self.cfg.ctbc.guide_lift_range
        lift = torch.where(stairs, (heights*self.cfg.ctbc.guide_lift_ratio).clamp(low, high),
                           torch.zeros_like(heights))
        count = self.ctbc_lift_table.shape[0]-1
        position = ((lift-low)/max(high-low, 1e-6)).clamp(0, 1)*count
        position = torch.where(stairs, position, torch.zeros_like(position))
        lower_i = position.floor().long().clamp(0, count)
        upper_i = (lower_i+1).clamp(0, count)
        fraction = (position-lower_i.float()).clamp(0, 1)[:, None]
        table = self.ctbc_lift_table
        self.guidance.set_amplitudes(
            env_ids, table[lower_i]*(1-fraction)+table[upper_i]*fraction)

    def set_training_iteration(self, iteration):
        """Cumulative PPO updates, so the anneal survives checkpoint loads."""
        self.ctbc_training_iteration = max(int(iteration), 0)

    @property
    def ctbc_feedforward_weight(self):
        cfg = self.cfg.ctbc
        if not cfg.enabled or cfg.pure_policy or not cfg.feedforward_enabled:
            return 0.0
        return self.guidance.feedforward_weight(self.ctbc_training_iteration)

    def ctbc_wheel_bottom(self):
        """Lowest wheel point. The Q1 wheel axis is Y, so spin keeps it a circle."""
        return self.rigid_body_states[:, self.feet_indices, 2]-self.cfg.asset.wheel_radius

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
        per_joint = getattr(self.cfg.control, 'leg_action_limits', None)
        leg_limit = self.cfg.normalization.leg_clip_actions
        if per_joint is not None:
            limits[[0, 1, 3, 4]] = torch.tensor(per_joint, device=limits.device, dtype=limits.dtype)
        elif leg_limit is not None:
            limits[[0, 1, 3, 4]] = leg_limit
        self.raw_action_clipping = (actions.to(self.device).abs() > limits.to(self.device)).float().mean(-1)
        if per_joint is not None:
            actions = actions.clone()
            actions[:, [0, 1, 3, 4]] = actions[:, [0, 1, 3, 4]].clamp(
                -limits[[0, 1, 3, 4]], limits[[0, 1, 3, 4]])
        elif leg_limit is not None:
            actions = actions.clone()
            actions[:, [0, 1, 3, 4]] = actions[:, [0, 1, 3, 4]].clamp(-leg_limit, leg_limit)
        cfg = self.cfg.ctbc
        if cfg.enabled and not cfg.pure_policy:
            force = self.contact_forces[:, self.feet_indices]
            self.guidance.update(force[..., :2].norm(dim=-1),
                                 force[..., 2] > self.cfg.rewards.support_force,
                                 self.ctbc_wheel_bottom(),
                                 self.ctbc_on_stairs
                                 & (self.commands[:, 0] > self.cfg.rewards.parking_speed_threshold))
        else:
            self.guidance.triggered.zero_()
            self.guidance.timed_out.zero_()
        self.ctbc_reference[:] = self.guidance.reference_offsets()
        weight = self.ctbc_feedforward_weight
        if weight:
            # Radians -> action units, then the per-joint clamp applies as usual.
            actions = actions.clone()
            leg = [0, 1, 3, 4]
            actions[:, leg] = actions[:, leg]+weight*self.ctbc_reference/self.ctbc_leg_scales
            actions[:, leg] = actions[:, leg].clamp(-limits[leg], limits[leg])
        return super().step(actions)

    def compute_reward(self):
        """Keep dt scaling; new runs leave negatively weighted penalties uncapped."""
        self.rew_buf.zero_()
        self.reward_terms = {}
        limit = self.cfg.rewards.clip_single_reward*self.dt
        for name, function in zip(self.reward_names, self.reward_functions):
            term = function()*self.reward_scales[name]
            if self.cfg.rewards.clip_constraint_rewards or self.reward_scales[name] >= 0:
                term = term.clamp(-limit, limit)
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
        adaptive_yaw = getattr(cfg, 'adaptive_yaw_tracking_sigma', {})
        if adaptive_yaw:
            sigma = command_tracking_sigma(self.commands[:, 1], sigma,
                cfg.parking_yaw_threshold, adaptive_yaw['full_speed'], adaptive_yaw['max_sigma'])
            return torch.exp(-difference.square()/sigma)
        mixture = getattr(cfg, 'yaw_tracking_mixture', {})
        if mixture:
            original = torch.exp(-difference.square()/sigma)
            mixed = positive_tracking_reward(
                difference, sigma, mixture['coarse_sigma'], mixture['fine_weight'])
            turning = self.commands[:, 1].abs() >= cfg.parking_yaw_threshold
            return torch.where(turning, mixed, original)
        if cfg.reward_version>=6:
            stationary=(self.commands[:,0].abs()<cfg.parking_speed_threshold)&(self.commands[:,1].abs()<cfg.parking_yaw_threshold)
            moving_sigma=torch.full_like(difference,sigma)
            if cfg.reward_version>=7:
                relative=cfg.relative_tracking
                width=(self.commands[:,1].abs()*relative['fraction']).clamp_min(relative['yaw_floor'])
                width=torch.where(self.commands[:,1].abs()<cfg.parking_yaw_threshold,
                                  torch.full_like(width,relative['zero_yaw_width']),width)
                moving_sigma=width.square()
            if cfg.reward_version>=8:
                settings=cfg.multi_scale_tracking
                dense=positive_tracking_reward(difference,moving_sigma,
                    settings['yaw_coarse_sigma'],settings['fine_weight'])
            else:
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
        adaptive = getattr(cfg, 'adaptive_tracking_sigma', {})
        if adaptive:
            moving_sigma = command_tracking_sigma(
                self.commands[:, 0], cfg.idle_tracking_sigma,
                cfg.parking_speed_threshold, adaptive['full_speed'], adaptive['max_sigma'])
        if cfg.reward_version>=7:
            relative=cfg.relative_tracking
            moving_sigma=(self.commands[:,0].abs()*relative['fraction']).clamp_min(relative['linear_floor']).square()
        sigma = torch.where(stationary, cfg.idle_tracking_sigma, moving_sigma)
        floor = getattr(cfg, 'turning_linear_sigma_floor', 0.0)
        if floor > 0:
            turning = self.commands[:, 1].abs() >= cfg.parking_yaw_threshold
            sigma = torch.where(turning, sigma.clamp_min(floor), sigma)
        if cfg.reward_version>=8:
            settings=cfg.multi_scale_tracking
            coarse_sigma=torch.where(stationary,
                torch.full_like(difference,settings['idle_linear_coarse_sigma']),
                torch.full_like(difference,settings['linear_coarse_sigma']))
            return positive_tracking_reward(difference,sigma,coarse_sigma,settings['fine_weight'])
        if cfg.reward_version>=6:
            return torch.where(stationary,torch.exp(-difference.square()/sigma),wide_tracking_reward(difference,sigma))
        return torch.exp(-difference.square()/sigma)

    def _reward_parking_drift(self):
        penalty = self.parking.penalty(self.cfg.rewards)
        quadratic_penalty = self.parking.penalty(self.cfg.rewards, quadratic=True)
        if getattr(self.cfg.rewards, 'parking_drift_stand_only', False):
            # Only the permanent stand group (lowest env ids); stops after motion are not penalized.
            stand_count = int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
            penalty = penalty*(torch.arange(self.num_envs, device=self.device) < stand_count)
            quadratic_penalty = quadratic_penalty*(torch.arange(self.num_envs, device=self.device) < stand_count)
        active = self.parking.weight > 0
        error = self.parking.error*active
        self.parking_tail_metrics = {
            'parking_tail_active_frames': active.sum(),
            'parking_tail_penalty_sum': penalty.sum(),
            'parking_tail_penalty_square_sum': penalty.square().sum(),
            'parking_tail_penalty_max': penalty.max(),
            'parking_tail_error_m_sum': error.sum(),
            'parking_tail_error_m_max': error.max(),
        }
        delta = float(getattr(self.cfg.rewards, 'parking_huber_delta_m', 0.))
        tail = active & ((self.parking.error-self.cfg.rewards.parking_deadband_m) > delta) & (delta > 0.)
        self.parking_tail_metrics.update({
            'parking_tail_quadratic_penalty_sum': quadratic_penalty.sum(),
            'parking_tail_quadratic_penalty_square_sum': quadratic_penalty.square().sum(),
            'parking_tail_quadratic_penalty_max': quadratic_penalty.max(),
            'parking_tail_huber_frames': tail.sum(),
            'parking_tail_penalty_saving_sum': (quadratic_penalty-penalty).sum(),
        })
        for label, threshold in [('2p5', .025), ('5', .05), ('10', .1),
                                 ('25', .25), ('50', .5), ('100', 1.), ('200', 2.)]:
            self.parking_tail_metrics['parking_tail_error_gt_'+label+'cm_frames'] = (active & (error > threshold)).sum()
        for threshold in (100, 1000):
            self.parking_tail_metrics['parking_tail_penalty_gt_'+str(threshold)+'_frames'] = (penalty > threshold).sum()
        return penalty

    def _leg_pose_error(self):
        # Use imported DOF indices; wheel phase never constrains the leg pose.
        error = self.dof_pos-self.default_dof_pos
        return error[:, self.leg_dof_indices].abs().sum(dim=1)

    def _reward_stand_still(self):
        settings = getattr(self.cfg.rewards, 'positive_stand_rewards', {})
        if settings:
            stationary = self.commands[:, :2].abs().amax(dim=1) <= settings['zero_command_tolerance']
            error = (self.dof_pos-self.default_dof_pos)[:, self.leg_dof_indices]
            return positive_stand_reward(error, settings['leg_width_rad'])*stationary
        stationary = self.commands[:, :2].norm(dim=1) < 0.1
        return self._leg_pose_error()*stationary

    def _reward_stand_wheel_still(self):
        settings = getattr(self.cfg.rewards, 'positive_stand_rewards', {})
        if settings:
            stationary = self.commands[:, :2].abs().amax(dim=1) <= settings['zero_command_tolerance']
            # Square each wheel separately: counter-rotation never cancels out.
            return positive_stand_reward(self.dof_vel[:, self.wheel_dof_indices],
                                         settings['wheel_width_rad_s'])*stationary
        stationary = self.commands[:, :2].norm(dim=1) < 0.1
        speed = self.dof_vel[:, self.wheel_dof_indices].abs()/31.0
        if self.cfg.rewards.clip_constraint_rewards:
            speed = speed.clamp(max=2.0)
        return speed.sum(dim=1)*stationary

    def _reward_ctbc_target(self):
        """Follow the teaching trajectory with the swinging leg."""
        actual = self.dof_pos[:, self.leg_dof_indices]
        target = self.default_dof_pos[:, self.leg_dof_indices]+self.ctbc_reference
        error = (actual-target).reshape(self.num_envs, 2, 2).square().mean(dim=2)
        reward = (torch.exp(-error/self.cfg.rewards.ctbc_target_sigma)*self.guidance.swing_mask).sum(dim=1)
        if self.cfg.rewards.ctbc_target_unscaled:
            # The reference trajectory outlives the feedforward injection: guidance
            # keeps running after the weight reaches zero, so the swing target is
            # still defined and the student can be rewarded for matching it.
            return reward
        initial = max(float(self.cfg.ctbc.feedforward_initial), 1e-6)
        return reward*min(self.ctbc_feedforward_weight/initial, 1.0)

    def _reward_ctbc_clearance(self):
        """The wheel bottom must really rise by the taught amount while swinging."""
        reference_q = (self.default_dof_pos[:, self.leg_dof_indices]
                       +self.ctbc_reference).reshape(self.num_envs, 2, 2)
        default_q = self.default_dof_pos[:, self.leg_dof_indices].reshape(self.num_envs, 2, 2)
        fk = lambda q: (-self.cfg.asset.l1*torch.cos(q[..., 0])
                        -self.cfg.asset.l2*torch.cos(q.sum(dim=-1)))
        desired = fk(reference_q)-fk(default_q)
        actual = self.ctbc_wheel_bottom()-self.guidance.start_height
        allowance = self.cfg.ctbc.clearance_overshoot_allowance
        error = (desired-actual).clamp_min(0)+(actual-desired-allowance).clamp_min(0)
        return (torch.exp(-error.square()/0.0009)*self.guidance.swing_mask).sum(dim=1)

    def _reward_ctbc_contact(self):
        """Keep the unswinging wheel loaded while the other leg is airborne."""
        swing = self.guidance.swing_mask
        support = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.support_force
        match = (support == ~swing).float().mean(dim=1)
        return match*(self.guidance.active_leg >= 0)*(self.guidance.phase >= 0.15)

    def _reward_nominal_state(self):
        """Leg-symmetry penalty, optionally waived while a CTBC swing lifts one leg:
        a one-leg lift is asymmetric by construction, so the penalty would veto it."""
        value = super()._reward_nominal_state()
        if self.cfg.rewards.nominal_state_swing_exempt:
            value = value*(~self.guidance.swing_mask.any(dim=1))
        return value

    def _reward_stair_rise(self):
        """Potential climb over stair tiles, paid as its positive increment.

        Height is the terrain under the base relative to that episode's spawn
        ground, so pitching cannot forge it. Non-stair tiles pay zero and a
        descent pays nothing; the cap bounds one episode's total potential.
        """
        ground = self._terrain_height_at(self.root_states[:, :2])
        rise = (ground-self.stair_spawn_ground).clamp_min(0.0)
        cap = self.cfg.rewards.stair_rise_cap
        progress = torch.where(self.stair_tile, rise.clamp_max(cap),
                               torch.zeros_like(rise))
        increment = (progress-self.stair_rise_prev).clamp(-cap, cap)
        self.stair_rise_prev.copy_(progress.detach())
        return increment

    def _reward_run_still(self):
        moving = self.commands[:, :2].norm(dim=1) > 0.1
        return self._leg_pose_error()*moving

    def _reward_dof_vel_limits(self):
        excess = (self.dof_vel.abs()
                  - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clamp_min(0)
        if self.cfg.rewards.clip_constraint_rewards:
            excess = excess.clamp(max=1.0)
        return excess.sum(dim=1)

    def _reward_parking_velocity(self):
        cfg = self.cfg.rewards
        speed = self.base_lin_vel[:, :2].norm(dim=-1)
        error = (speed-cfg.parking_velocity_deadband).clamp_min(0)
        fine=error/cfg.parking_velocity_scale
        if cfg.clip_constraint_rewards:
            fine=fine.clamp(max=4)
        if cfg.reward_version>=9:
            settings=cfg.multi_scale_parking
            coarse=error/settings['velocity_scale']
            if cfg.clip_constraint_rewards:
                coarse=coarse.clamp(max=4)
            fine=settings['fine_weight']*fine+(1-settings['fine_weight'])*coarse
        return self.parking.weight*fine

    def _reward_termination(self):
        # Simultaneous timeout must not hide an actual fall in reward v2.
        if self.cfg.rewards.reward_version >= 2:
            return self.physical_failure.float()
        return super()._reward_termination()

    def _resample_commands(self, env_ids):
        if not getattr(self, "external_commands", False):
            selection = super()._resample_commands(env_ids)
            axes = self.cfg.commands.axis_sampling
            if axes:
                apply_axis_sampling(self.commands, env_ids, selection,
                    self.cfg.commands.stand_probability, axes,
                    self.cfg.rewards.parking_speed_threshold, self.cfg.rewards.parking_yaw_threshold)
            self._apply_sequence_commands(env_ids)
            fraction = getattr(self.cfg.commands, 'climb_command_fraction', 0.0)
            if fraction > 0 and len(env_ids):
                stand_count = int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
                chosen = (self._climb_tile_mask(env_ids) & (env_ids >= stand_count)
                          & (torch.rand(len(env_ids), device=self.device) < fraction))
                ids = env_ids[chosen]
                low, high = self.cfg.commands.climb_command_vx
                self.commands[ids, 0] = low+(high-low)*torch.rand(len(ids), device=self.device)
                self.commands[ids, 1] = 0.0
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
        if getattr(self, 'external_commands', False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        stand_count=int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
        stand_ids=env_ids[env_ids<stand_count]
        self.commands[stand_ids,:2]=0.
        parking=self.cfg.commands.parking_sequence
        parking_end=stand_count
        if parking:
            parking_end+=int(self.num_envs*parking['fraction'])
            ids=env_ids[(env_ids>=stand_count) & (env_ids<parking_end)]
            time_s=self.episode_length_buf[ids].float()*self.dt
            sign=1.-2.*(ids%2).float()
            self.commands[ids,0]=parking_sequence_velocity(time_s,parking['speed'],sign)
            self.commands[ids,1]=0.
        if fraction<=0:
            return
        ids=env_ids[(env_ids<int(self.num_envs*fraction)) & (env_ids>=parking_end)]
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

    def _apply_standing_episode_timeout(self):
        """Extend only the permanent standing group; physical falls still reset."""
        duration = self.cfg.env.permanent_stand_episode_length_s
        count = int(self.num_envs*self.cfg.commands.permanent_stand_fraction)
        # All evaluation scenario builders set permanent_stand_fraction=0.
        if duration == 0 or count == 0 or getattr(self, 'evaluation_mode', False):
            return
        if not math.isfinite(duration) or duration < self.cfg.env.episode_length_s:
            raise ValueError('Standing horizon must be finite and no shorter than normal episodes')
        steps = math.ceil(duration/self.dt)
        self.time_out_buf[:count] = self.episode_length_buf[:count] >= steps
        self.reset_buf[:count] = self.physical_failure[:count] | self.time_out_buf[:count]

    def check_termination(self):
        super().check_termination()
        physical_failure = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices], dim=-1) > 5.0, dim=1)
        physical_failure |= self.projected_gravity[:, 2] > -math.cos(self.cfg.env.max_tilt_rad)
        physical_failure |= self.base_height < self.cfg.env.min_base_height
        self.physical_failure = physical_failure
        self._apply_standing_episode_timeout()
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
        slope, clearance = self._terrain_labels()
        self.estimator_targets[:] = torch.cat((
            self.base_lin_vel/cfg.velocity_scale, self.wheel_contact.float(),
            slip/cfg.slip_scale, torch.ones(self.num_envs, 1, device=self.device),
            support.float(), slope*SLOPE_LABEL_SCALE, clearance*CLEARANCE_LABEL_SCALE,
            self._step_labels()*CLEARANCE_LABEL_SCALE), dim=-1)
        if getattr(self, "custom_origins", False):
            distance = (self.root_states[:, :2]-self.env_origins[:, :2]).norm(dim=-1)
            self.max_origin_distance[:] = torch.maximum(self.max_origin_distance, distance)
            if getattr(self.cfg.terrain, 'climb_curriculum', False):
                ground = self._terrain_height_at(self.root_states[:, :2])
                self.climb_max_rise[:] = torch.maximum(self.climb_max_rise, ground-self.stair_spawn_ground)
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
            "parking_weight": self.parking.weight.clone(),
            "parking_error": self.parking.error.clone(),
            "braking_distance": self.parking.braking_distance.clone(),
            "action_clipping": self.raw_action_clipping.clone(),
            "ctbc_swing": self.guidance.swing_mask.any(dim=-1).clone(),
            # Stage 0 range-usage diagnostics; thresholds live in the evaluation standard.
            "action": self.actions.clone(),
            "base_height": self.base_height.clone(),
        }
        self.previous_world_velocity[:] = self.root_states[:, 7:10]
        self.terminal_critic_obs[:] = self._critic_observations(self.compute_proprioception_observations())

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # Match loco_gym T0/T1 reset randomization. Defaults to zero so old
        # checkpoints and fixed evaluation scenarios retain their protocol.
        radius = getattr(self.cfg.terrain, "spawn_xy_range", 0.0)
        if self.custom_origins and radius > 0 and len(env_ids):
            self.root_states[env_ids, :2] += (
                2 * torch.rand(len(env_ids), 2, device=self.device) - 1) * radius
            ids = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(ids), len(ids))

        # Audit the first full reset before the simulator advances it.
        if len(env_ids) == self.num_envs and not hasattr(self, '_spawn_audit_xy'):
            self._spawn_audit_xy = self.root_states[:, :2].clone()
            self._spawn_audit_origins_xy = self.env_origins[:, :2].clone()

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
            self.max_origin_distance[env_ids] = 0
            # A plane has no terrain object and no stairs.
            risers = getattr(getattr(self, 'terrain', None), 'riser_heights', None)
            if risers is None:
                self.stair_tile[env_ids] = False
            else:
                rows = self.terrain_levels[env_ids].cpu().numpy().clip(0, risers.shape[0]-1)
                cols = self.terrain_types[env_ids].cpu().numpy().clip(0, risers.shape[1]-1)
                self.stair_tile[env_ids] = torch.as_tensor(
                    risers[rows, cols] > 1e-6, device=self.device, dtype=torch.bool)
            self.stair_spawn_ground[env_ids] = self.env_origins[env_ids, 2]
            self.stair_rise_prev[env_ids] = 0
            self.climb_max_rise[env_ids] = 0
            self.history_reset[env_ids] = True
            self.pulse_force[env_ids] = 0
            self.eval_force[env_ids] = 0
            self.parking.reset(env_ids, self.root_states[:, :3])
            self.guidance.reset(env_ids)
            self._refresh_guide_amplitudes(env_ids)
