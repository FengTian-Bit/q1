"""CPU interface tests using real task methods and a tensor-only Gym stub.

This does NOT simulate contacts or establish that Q1 balances. Run from the repo
root: python -m unittest discover -s tests -p test_q1_flat.py -v
Requires numpy and torch, but no Isaac Gym or pytest.
"""
import ast
import builtins
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

ROOT = Path(os.environ.get("Q1_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
OVERLAY = Path(os.environ.get("Q1_OVERLAY_ROOT", ROOT))


def source(path):
    candidate = OVERLAY / path
    return candidate if candidate.exists() else ROOT / path


def load_code(path, namespace):
    """Execute actual class/function bodies without importing Isaac Gym."""
    tree = ast.parse(source(path).read_text(encoding="utf-8"), filename=path)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(source(path)), "exec"), namespace)
    return namespace


def class_to_dict(obj):
    if isinstance(obj, (str, float, int, bool, list, tuple, dict)) or obj is None:
        return obj
    return {key: class_to_dict(getattr(obj, key)) for key in dir(obj)
            if not key.startswith("_") and not callable(getattr(obj, key))}


def quat_rotate_inverse(q, v):
    xyz, w = q[:, :3], q[:, 3:4]
    return v * (2 * w.square() - 1) - 2 * w * torch.cross(xyz, v, dim=-1) + 2 * xyz * (xyz * v).sum(-1, keepdim=True)


NS = dict(torch=torch, np=np, nn=nn, Normal=Normal, math=math, inspect=inspect,
          Tensor=Tensor, Dict=Dict, Tuple=Tuple, BaseTask=object,
          class_to_dict=class_to_dict, quat_rotate_inverse=quat_rotate_inverse,
          to_torch=lambda x, **kw: torch.tensor(x, dtype=torch.float32, **kw),
          get_axis_params=lambda value, axis: [value if i == axis else 0.0 for i in range(3)],
          torch_rand_float=lambda low, high, shape, device: low + (high-low) * torch.rand(shape, device=device),
          gymtorch=SimpleNamespace(wrap_tensor=lambda x: x, unwrap_tensor=lambda x: x))
load_code("wheel_legged_gym/envs/base/base_config.py", NS)
load_code("wheel_legged_gym/envs/base/legged_robot_config.py", NS)
load_code("wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py", NS)
load_code("wheel_legged_gym/envs/q1_flat/q1_flat_config.py", NS)
load_code("wheel_legged_gym/envs/base/legged_robot.py", NS)
load_code("wheel_legged_gym/envs/q1_flat/q1_flat.py", NS)
Q1Flat, Q1FlatCfg = NS["Q1Flat"], NS["Q1FlatCfg"]
def quat_apply(q, v):
    cross = torch.cross(q[:, :3], v, dim=-1)
    return v + 2*(q[:, 3:4]*cross + torch.cross(q[:, :3], cross, dim=-1))
NS["quat_apply"] = quat_apply
load_code("wheel_legged_gym/envs/q1/q1_config.py", NS)
load_code("wheel_legged_gym/utils/terrain.py", NS)
load_code("wheel_legged_gym/envs/q1/q1.py", NS)


class TensorGym:
    def __init__(self, count):
        self.root = torch.zeros(count, 13)
        self.root[:, 2], self.root[:, 6] = 0.402, 1.0
        self.dof = torch.zeros(count * 6, 2)
        self.contact = torch.zeros(count * 7, 3)
        self.rigid = torch.zeros(count, 7, 13)
        self.rigid[:, :, 6] = 1.0
        self.rigid[:, [3, 6], 2] = 0.075

    def acquire_rigid_body_state_tensor(self, sim):
        return self.rigid

    def get_actor_rigid_body_properties(self, env, actor):
        return [SimpleNamespace(com=SimpleNamespace(x=0.0, y=0.0, z=0.0)) for _ in range(7)]

    def acquire_actor_root_state_tensor(self, sim):
        return self.root

    def acquire_dof_state_tensor(self, sim):
        return self.dof

    def acquire_net_contact_force_tensor(self, sim):
        return self.contact

    def __getattr__(self, name):
        if name.startswith(("refresh_", "set_")) or name in ("simulate", "fetch_results"):
            return lambda *args: None
        raise AttributeError(name)


def make_env(count=32, permutation=None, max_delay_ms=0, task_class=Q1Flat, cfg_class=Q1FlatCfg):
    cfg = cfg_class()
    cfg.env.num_envs = count
    # Existing behavioral tests explicitly use a plane. Terrain integration
    # tests attach real height arrays and enable the rough-terrain settings.
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    cfg.domain_rand.delay_ms_range = [0, max_delay_ms]
    cfg.domain_rand.randomize_action_delay = max_delay_ms > 0
    env = task_class.__new__(task_class)
    env.cfg, env.device, env.sim = cfg, "cpu", None
    env.sim_params = SimpleNamespace(dt=cfg.sim.dt)
    env.envs, env.actor_handles = [None], [None]
    env.num_envs, env.num_obs, env.num_actions = count, cfg.env.num_observations, 6
    env.num_privileged_obs, env.obs_history_length = cfg.env.num_privileged_obs, 5
    env.num_dof = env.num_dofs = 6
    env.num_bodies, env.up_axis_idx = 7, 2
    names = cfg.asset.policy_joint_names
    env.dof_names = list(names) if permutation is None else [names[i] for i in permutation]
    env.policy_dof_indices = torch.tensor([env.dof_names.index(name) for name in names])
    env.leg_dof_indices = env.policy_dof_indices[[0, 1, 3, 4]]
    env.wheel_dof_indices = env.policy_dof_indices[[2, 5]]
    env.feet_indices = torch.tensor([3, 6])
    env.termination_contact_indices = torch.tensor([0])
    env.penalised_contact_indices = torch.tensor([0, 1, 2, 4, 5])
    env.gym, env.pi = TensorGym(count), torch.tensor(math.pi)
    env.env_origins = torch.zeros(count, 3)
    env.base_init_state = torch.tensor(cfg.init_state.pos + cfg.init_state.rot + [0.0]*6)
    env.base_mass = torch.full((count,), 4.13466807)
    env.base_com = torch.zeros(count, 3)
    env.friction_coef = torch.full((count,), 0.8)
    env.restitution_coef = torch.zeros(count)
    env.obs_buf, env.privileged_obs_buf = torch.zeros(count, env.num_obs), torch.zeros(count, env.num_privileged_obs)
    env.obs_history = torch.zeros(count, env.num_obs*5)
    env.rew_buf = torch.zeros(count)
    for name in ("reset_buf", "time_out_buf", "edge_reset_buf"):
        setattr(env, name, torch.zeros(count, dtype=torch.bool))
    for name in ("episode_length_buf", "envs_steps_buf", "fail_buf"):
        setattr(env, name, torch.zeros(count, dtype=torch.long))
    env.viewer, env.init_done, env.custom_origins = None, True, False
    env.render = lambda: None
    env._parse_cfg(cfg)
    env._init_buffers()
    urdf = ET.parse(source("resources/robots/q1/urdf/q1.urdf")).getroot()
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    env.torque_limits = torch.tensor([float(joints[name].find("limit").get("effort")) for name in env.dof_names])
    env.dof_pos_limits = torch.tensor([[float(joints[name].find("limit").get("lower")),
                                      float(joints[name].find("limit").get("upper"))]
                                     for name in env.dof_names])
    env.dof_vel_limits = torch.tensor([float(joints[name].find("limit").get("velocity")) for name in env.dof_names])
    env._prepare_reward_function()
    env.reset_idx(torch.arange(count))
    env._update_leg_kinematics()
    env._post_physics_step_callback()
    env.compute_observations()
    return env


class Q1Tests(unittest.TestCase):
    def test_model_axes_limits_and_resources(self):
        model = ET.parse(source("resources/robots/q1/urdf/q1.urdf")).getroot()
        active = [j for j in model.findall("joint") if j.get("type") != "fixed"]
        self.assertEqual(set(j.get("name") for j in active), set(Q1FlatCfg.asset.policy_joint_names))
        for joint in active:
            self.assertEqual(joint.find("axis").get("xyz"), "0 1 0")
            if joint.get("name").endswith("foot_joint"):
                self.assertEqual(joint.get("type"), "revolute")
                self.assertEqual(float(joint.find("limit").get("lower")), -99999)
                self.assertEqual(float(joint.find("limit").get("upper")), 99999)
        original_urdf = ROOT / "resources/robots/q1/urdf/q1.urdf"
        for mesh in model.findall(".//mesh"):
            self.assertTrue((original_urdf.parent / mesh.get("filename")).is_file())

    def test_same_physical_control_under_importer_permutation(self):
        normal = make_env()
        shuffled = make_env(permutation=[5, 2, 4, 0, 3, 1])
        actions = torch.tensor([[0.2, -0.2, 1.0, 0.2, -0.2, 1.0]]).repeat(normal.num_envs, 1)
        actual = normal._compute_torques(actions)[:, normal.policy_dof_indices]
        permuted = shuffled._compute_torques(actions)[:, shuffled.policy_dof_indices]
        torch.testing.assert_close(actual, permuted)
        torch.testing.assert_close(actual[:, [0, 1]], actual[:, [3, 4]])
        torch.testing.assert_close(actual[:, 2], actual[:, 5])
        self.assertTrue((actual[:, [2, 5]] > 0).all())
        saturated = normal._compute_torques(torch.full_like(actions, 100.0))
        self.assertTrue((saturated.abs() <= normal.torque_limits).all())

    def test_wheel_phase_does_not_change_actor_observation(self):
        env = make_env()
        before = env.compute_proprioception_observations()
        env.dof_pos[:, env.wheel_dof_indices] = 20000.0
        torch.testing.assert_close(before, env.compute_proprioception_observations())
        self.assertEqual(before.shape, (32, 27))
        torch.testing.assert_close(env.theta0, torch.zeros(32, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(env.L0, torch.full((32, 2), 0.4*math.cos(0.6)))

    def test_noise_layout_keeps_commands_actions_and_wheel_phase_clean(self):
        env = make_env()
        env.cfg.noise.add_noise = True
        noise = env._get_noise_scale_vec(env.cfg)
        self.assertTrue((noise[6:9] == 0).all())
        self.assertTrue((noise[21:27] == 0).all())
        self.assertEqual(noise[11].item(), 0.0)
        self.assertEqual(noise[14].item(), 0.0)
        self.assertTrue((noise[15:21] > 0).all())

    def test_reset_clears_action_history_and_tilt(self):
        env = make_env()
        env.actions[:] = 0.9
        env.action_fifo[:] = 0.9
        env.last_actions[:] = 0.9
        env.root_states[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        env.projected_gravity[:, 2] = 1.0
        env.reset_idx(torch.arange(env.num_envs))
        self.assertEqual(env.actions.abs().sum().item(), 0)
        self.assertEqual(env.action_fifo.abs().sum().item(), 0)
        self.assertEqual(env.last_actions.abs().sum().item(), 0)
        self.assertEqual(env.obs_history[:, -6:].abs().sum().item(), 0)
        self.assertTrue((env.projected_gravity[:, 2] == -1).all())

    def test_standing_and_low_speed_commands(self):
        env = make_env(1024)
        torch.manual_seed(7)
        env._resample_commands(torch.arange(env.num_envs))
        speed = env.commands[:, 0]
        self.assertTrue((speed.abs() <= 0.3).all())
        self.assertTrue((speed > 0).any() and (speed < 0).any())
        fraction = (speed == 0).float().mean().item()
        self.assertTrue(0.4 < fraction < 0.6)
        self.assertTrue((env.commands[:, 1] == 0).all())
        env.cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env.command_ranges["lin_vel_x"][:] = 0.0
        env.cfg.commands.stand_probability = 1.0
        env._resample_commands(torch.arange(env.num_envs))
        self.assertEqual(env.commands[:, :2].abs().sum().item(), 0)

    def test_delay_maximum_is_in_bounds(self):
        env = make_env(64, max_delay_ms=10)
        self.assertEqual(env.action_fifo.shape[1], 3)
        env.action_delay_idx[:] = 2
        result = env.step(torch.ones(64, 6))
        self.assertEqual(result[0].shape, (64, 27))
        self.assertEqual(result[1].shape, (64, 143))
        self.assertTrue(torch.isfinite(result[1]).all())
        self.assertEqual(result[5].shape, (64, 135))

    def test_failures_terminate_and_timeouts_are_distinct(self):
        env = make_env()
        env.contact_forces[0, 0, 2] = 10.0
        env.projected_gravity[1, 2] = 0.0
        env.base_height[2] = 0.1
        env.episode_length_buf[3] = env.max_episode_length
        env.check_termination()
        self.assertTrue(env.reset_buf[:4].all())
        self.assertFalse(env.reset_buf[4:].any())
        self.assertFalse(env.time_out_buf[:3].any())
        self.assertTrue(env.time_out_buf[3])
        torch.testing.assert_close(env._reward_alive()[:4], torch.tensor([0., 0., 0., 1.]))

    def test_registration_preserves_original_tasks(self):
        registered = {}
        namespace = dict(NS)
        load_code("wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc_config.py", namespace)
        load_code("wheel_legged_gym/envs/wheel_legged_vmc_flat/wheel_legged_vmc_flat_config.py", namespace)
        load_code("wheel_legged_gym/envs/q1_ctbc/q1_ctbc_config.py", namespace)
        load_code("wheel_legged_gym/envs/q1_ctbc/q1_ctbc.py", namespace)
        namespace["deepcopy"] = __import__("copy").deepcopy
        load_code("wheel_legged_gym/envs/q1/q1_t0_config.py", namespace)
        load_code("wheel_legged_gym/envs/q1/q1_t0.py", namespace)
        namespace["LeggedRobotVMC"] = object
        namespace["task_registry"] = SimpleNamespace(register=lambda name, cls, cfg, train: registered.update({name: (cls, cfg, train)}))
        load_code("wheel_legged_gym/envs/__init__.py", namespace)
        self.assertEqual(set(registered), {"wheel_legged", "wheel_legged_vmc", "wheel_legged_vmc_flat", "q1_flat", "q1", "q1_ctbc"})
        self.assertIs(registered["q1_flat"][0], Q1Flat)
        self.assertIs(registered["q1"][0], namespace["Q1T0CTBC"])
        self.assertTrue(issubclass(registered["q1"][0], namespace["Q1CTBC"]))
        self.assertEqual(registered["q1"][1].terrain.generation_mode, "t0")
        self.assertIs(registered["q1_ctbc"][0], namespace["Q1CTBC"])
        self.assertEqual(registered["wheel_legged"][1].asset.name, "WheelLegged")

    def test_playback_runs_across_inference_mode_frames(self):
        env = make_env(2)
        train_cfg = NS["Q1FlatCfgPPO"]()
        runner = SimpleNamespace(
            alg=SimpleNamespace(actor_critic=SimpleNamespace(is_sequence=False)),
            get_inference_policy=lambda device: lambda obs: torch.zeros(2, 6)
        )
        registry = SimpleNamespace(
            get_cfgs=lambda task: (env.cfg, train_cfg),
            make_env=lambda *args, **kwargs: (env, env.cfg),
            make_alg_runner=lambda *args, **kwargs: (runner, train_cfg)
        )
        namespace = dict(NS, os=os, task_registry=registry,
                         WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT),
                         export_policy_as_jit=lambda *args: None,
                         range=lambda count: builtins.range(min(count, 3)))
        load_code("wheel_legged_gym/envs/q1_flat/play_q1.py", namespace)
        namespace["play_q1"](SimpleNamespace(task="q1_flat", num_envs=2, q1_stage="balance"))
        self.assertTrue(torch.isfinite(env.obs_buf).all())

    def test_real_ppo_rollout_and_update_with_tensor_stub(self):
        # Run the repository's actual encoder, storage and PPO code, using
        # tensor-only steps. This checks interfaces/backprop, not dynamics.
        from torch import optim
        import statistics
        from collections import deque
        namespace = dict(NS, optim=optim, statistics=statistics, deque=deque)
        load_code("wheel_legged_gym/rsl_rl/modules/actor_critic.py", namespace)
        load_code("wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py", namespace)
        load_code("wheel_legged_gym/rsl_rl/storage/rollout_storage.py", namespace)
        load_code("wheel_legged_gym/rsl_rl/algorithms/ppo.py", namespace)
        env = make_env(8)
        model = namespace["ActorCriticSequence"](
            27, 146, 6, 135, 3, encoder_hidden_dims=[16, 8],
            actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8], init_noise_std=0.5
        )
        ppo = namespace["PPO"](model, num_learning_epochs=1, num_mini_batches=2, desired_kl=0.01)
        ppo.init_storage(8, 4, [27], [146], [135], [6])
        for _ in range(2):
            with torch.inference_mode():
                for _ in range(4):
                    actions = ppo.act(env.obs_buf, env.obs_history, env.privileged_obs_buf)
                    obs, critic, reward, done, info, history = env.step(actions)
                    ppo.process_env_step(reward, done, info, obs)
                ppo.compute_returns(torch.cat((critic, model.encode(history)), dim=-1))
            losses = ppo.update()
            self.assertTrue(all(math.isfinite(value) for value in losses))


if __name__ == "__main__":
    unittest.main(verbosity=2)
