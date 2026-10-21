"""Q1-specific mapping and flat-ground resets on the wheel_legged PD backend."""

import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


class Q1Flat(LeggedRobot):
    def _create_envs(self):
        super()._create_envs()
        names = self.cfg.asset.policy_joint_names
        if len(self.dof_names) != 6 or set(self.dof_names) != set(names):
            raise ValueError("Q1 expected six named DOFs, got {}".format(self.dof_names))
        self.policy_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in names], device=self.device
        )
        self.leg_dof_indices = self.policy_dof_indices[[0, 1, 3, 4]]
        self.wheel_dof_indices = self.policy_dof_indices[[2, 5]]
        # Exact wheel bodies: never treat the fixed motor housings as wheels.
        handles = [self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], name
        ) for name in self.cfg.asset.wheel_body_names]
        if any(handle < 0 for handle in handles):
            raise ValueError("Q1 wheel bodies were not imported: {}".format(handles))
        self.feet_indices = torch.tensor(handles, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        # Delay index includes zero (the current action), hence max_delay + 1.
        max_delay = int(math.ceil(
            self.cfg.domain_rand.delay_ms_range[1] / 1000.0 / self.sim_params.dt
        ))
        self.action_fifo = torch.zeros(
            self.num_envs, max_delay + 1, self.num_actions, device=self.device
        )

    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        for prop in props:
            if not self.cfg.domain_rand.randomize_friction:
                prop.friction = self.cfg.terrain.static_friction
            if not self.cfg.domain_rand.randomize_restitution:
                prop.restitution = self.cfg.terrain.restitution
        if not self.cfg.domain_rand.randomize_friction:
            self.friction_coef[env_id] = self.cfg.terrain.static_friction
        if not self.cfg.domain_rand.randomize_restitution:
            self.restitution_coef[env_id] = self.cfg.terrain.restitution
        return props

    def _process_rigid_body_props(self, props, env_id):
        # The fixed COM is still useful privileged information when DR is off.
        props = super()._process_rigid_body_props(props, env_id)
        if not self.cfg.domain_rand.randomize_base_com:
            self.base_com[env_id] = torch.tensor(
                [props[0].com.x, props[0].com.y, props[0].com.z], device=self.device
            )
        return props

    def _compute_torques(self, actions):
        position_ref = torch.zeros_like(self.dof_pos)
        velocity_ref = torch.zeros_like(self.dof_vel)
        position_ref[:, self.leg_dof_indices] = (
            actions[:, [0, 1, 3, 4]] * self.cfg.control.pos_action_scale
        )
        velocity_ref[:, self.wheel_dof_indices] = (
            actions[:, [2, 5]] * self.cfg.control.vel_action_scale
        )
        torque = self.p_gains * (
            position_ref + self.default_dof_pos - self.dof_pos
        ) + self.d_gains * (velocity_ref - self.dof_vel)
        return torch.clip(torque * self.torques_scale, -self.torque_limits, self.torque_limits)

    def compute_proprioception_observations(self):
        position_error = (
            self.dof_pos - self.default_dof_pos
        )[:, self.policy_dof_indices].clone()
        # Wheel phase has no control meaning on this round-wheel robot.
        position_error[:, [2, 5]] = 0.0
        return torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            position_error * self.obs_scales.dof_pos,
            self.dof_vel[:, self.policy_dof_indices] * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        noise = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        scales, level = cfg.noise.noise_scales, cfg.noise.noise_level
        noise[0:3] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise[3:6] = scales.gravity * level
        noise[9:15] = scales.dof_pos * level * self.obs_scales.dof_pos
        noise[[11, 14]] = 0.0
        noise[15:21] = scales.dof_vel * level * self.obs_scales.dof_vel
        return noise

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        stand = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.stand_probability
        self.commands[env_ids[stand], 0:2] = 0.0

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        lin = self.cfg.init_state.reset_lin_vel
        ang = self.cfg.init_state.reset_ang_vel
        self.root_states[env_ids, 7:9] += torch_rand_float(
            -lin, lin, (len(env_ids), 2), device=self.device
        )
        self.root_states[env_ids, 10:12] += torch_rand_float(
            -ang, ang, (len(env_ids), 2), device=self.device
        )
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )
        self.base_height[env_ids] = self.root_states[env_ids, 2] - self.env_origins[env_ids, 2]
        ids = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids), len(ids)
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # No delayed command or action observation may leak across episodes.
        self.actions[env_ids] = 0.0
        self.action_fifo[env_ids] = 0.0
        self.dof_acc[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        super().reset_idx(env_ids)

    def _update_leg_kinematics(self):
        q = self.dof_pos[:, self.leg_dof_indices].reshape(self.num_envs, 2, 2)
        thigh, calf = q[:, :, 0], q[:, :, 1]
        # Q1 zero pose points down -Z; all six actuated axes are +Y.
        end_x = -self.cfg.asset.l1 * torch.sin(thigh) - self.cfg.asset.l2 * torch.sin(thigh + calf)
        end_z = -self.cfg.asset.l1 * torch.cos(thigh) - self.cfg.asset.l2 * torch.cos(thigh + calf)
        self.L0 = torch.sqrt(end_x.square() + end_z.square())
        self.theta0 = torch.atan2(end_x, -end_z)

    def post_physics_step(self):
        # Infos belong to one transition. Reusing the last reset's dictionary
        # leaks timeout bootstrapping and repeats episode logs on later steps.
        self.extras = {}
        # The inherited version hard-codes the wl robot's right-leg signs and
        # knee zero offset, so update Q1 kinematics explicitly here.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        self._update_leg_kinematics()
        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()
        self.compute_observations()
        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions
        self.last_base_position[:] = self.base_position
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        contact = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 5.0, dim=1)
        fallen = self.projected_gravity[:, 2] > -math.cos(self.cfg.env.max_tilt_rad)
        fallen |= self.base_height < self.cfg.env.min_base_height
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = contact | fallen | self.time_out_buf

    def _reward_alive(self):
        return (~(self.reset_buf & ~self.time_out_buf)).float()

    def _reward_lateral_velocity(self):
        return self.base_lin_vel[:, 1].square()

    def _reward_base_height(self):
        return torch.exp(-(self.base_height - self.commands[:, 2]).square() / self.cfg.rewards.height_sigma)

    def _reward_dof_vel(self):
        return self.dof_vel[:, self.leg_dof_indices].square().sum(dim=1)

    def _reward_action_smooth(self):
        delta = self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]
        return delta[:, [0, 1, 3, 4]].square().sum(dim=1)

    def _reward_dof_pos_limits(self):
        position = self.dof_pos[:, self.leg_dof_indices]
        limits = self.dof_pos_limits[self.leg_dof_indices]
        return ((limits[:, 0] - position).clamp(min=0.0)
                + (position - limits[:, 1]).clamp(min=0.0)).sum(dim=1)
