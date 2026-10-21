"""Terrain API, map layout and tensor behavior tests, without PhysX.

The primitive spy records the arguments sent to Isaac Gym. It does not claim
to reproduce Gym geometry. Surface/normal tests use independent analytic maps.
"""
from copy import deepcopy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_q1_flat import NS, make_env, load_code


def config():
    return deepcopy(NS["Q1Cfg"]().terrain)


class PrimitiveSpy:
    def __init__(self):
        self.calls = []

    def SubTerrain(self, name, width, length, vertical_scale, horizontal_scale):
        return SimpleNamespace(name=name, width=width, length=length,
                               vertical_scale=vertical_scale, horizontal_scale=horizontal_scale,
                               height_field_raw=np.zeros((width, length), dtype=np.int16))

    def __getattr__(self, name):
        def record(terrain, *args, **kwargs):
            self.calls.append((name, terrain.name, args, kwargs))
            # A unique nonzero flat surface exercises assembly/origin logic.
            terrain.height_field_raw[:] = len(self.calls)
        return record

    def convert_heightfield_to_trimesh(self, heights, horizontal_scale, vertical_scale, slope_threshold):
        self.mesh_call = (heights.shape, horizontal_scale, vertical_scale, slope_threshold)
        return np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32)


def rough_env(count=4, slope=0.0, height=0.0):
    cfg = NS["Q1Cfg"]()
    NS["configure_q1_stage"](cfg, "low_speed")
    env = make_env(count, task_class=NS["Q1"], cfg_class=lambda: cfg)
    cfg = env.cfg.terrain
    cfg.mesh_type, cfg.curriculum = "heightfield", False
    cfg.border_size, cfg.horizontal_scale, cfg.vertical_scale = 0.0, 0.1, 0.005
    cfg.num_rows, cfg.num_cols = 3, 2
    cfg.max_init_terrain_level = 1
    cfg.terrain_width, cfg.terrain_length = 6.0, 8.0
    x = torch.arange(260).float()[:, None] * cfg.horizontal_scale
    env.height_samples = torch.round((slope*x + height) / cfg.vertical_scale).short().expand(260, 140).clone()
    env.root_states[:, 0:2] = torch.tensor([4.0, 3.0])
    ground = slope*4.0 + height
    env.root_states[:, 2] = ground + 0.402
    env.rigid_body_states[:, env.feet_indices, 0] = 4.0
    env.rigid_body_states[:, env.feet_indices, 1] = 3.0
    env.rigid_body_states[:, env.feet_indices, 2] = ground + 0.075
    env.env_class = torch.zeros(count, dtype=torch.long)
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    env.terrain_class_map = torch.tensor([[-1, 2], [-1, 2], [-1, 2]])
    env.terrain = SimpleNamespace(env_origins=np.array([
        [[4.0, 3.0, height], [4.0, 9.0, height]],
        [[12.0, 3.0, height+0.1], [12.0, 9.0, height+0.1]],
        [[20.0, 3.0, height+0.2], [20.0, 9.0, height+0.2]],
    ]), terrain_classes=env.terrain_class_map.numpy())
    return env


class Q1TerrainTests(unittest.TestCase):
    def test_formal_terrain_defaults_and_severity(self):
        cfg = config()
        self.assertEqual((cfg.mesh_type, cfg.curriculum), ("trimesh", True))
        self.assertEqual((cfg.num_rows, cfg.num_cols), (20, 20))
        self.assertEqual((cfg.terrain_length, cfg.terrain_width), (8.0, 8.0))
        self.assertAlmostEqual(sum(cfg.terrain_proportions), 1.0)
        self.assertEqual(cfg.slope_range, [0.05, 0.35])
        self.assertEqual(cfg.rough_slope_range, [0.05, 0.25])
        self.assertEqual(cfg.rough_height_range, [-0.025, 0.025])
        self.assertEqual(cfg.stair_height_range, [0.025, 0.20])
        self.assertEqual(cfg.obstacle_height_range, [0.025, 0.15])
        self.assertEqual(cfg.stair_width, 0.25)
        self.assertEqual((len(cfg.measured_points_x), len(cfg.measured_points_y)), (11, 7))
        self.assertFalse(NS["Q1FlatCfg"]().terrain.curriculum)

    def test_all_tile_types_exact_endpoints_and_assembly(self):
        cfg = config()
        cfg.mesh_type, cfg.num_rows, cfg.border_size = "heightfield", 2, 0.0
        # Rectangular tiles catch width/length transposition.
        cfg.terrain_width = 6.0
        spy = PrimitiveSpy()
        with patch.dict(NS, terrain_utils=spy):
            terrain = NS["Q1Terrain"](cfg, 4096)
        self.assertEqual(terrain.height_field_raw.shape, (160, 1200))
        self.assertEqual(terrain.terrain_classes[0].tolist(),
                         [-1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
        stairs = [kw["step_height"] for kind, _, _, kw in spy.calls if kind == "pyramid_stairs_terrain"]
        self.assertEqual(set(round(h, 4) for h in stairs), {-0.20, -0.025, 0.025, 0.20})
        slope = [kw["slope"] for kind, _, _, kw in spy.calls if kind == "pyramid_sloped_terrain"]
        self.assertEqual(set(round(s, 4) for s in slope), {-0.35, -0.05, 0.05, 0.25, 0.35})
        noise = [kw for kind, _, _, kw in spy.calls if kind == "random_uniform_terrain"]
        self.assertTrue(all(kw["min_height"] == -0.025 and kw["max_height"] == 0.025 for kw in noise))
        obstacles = [args[0] for kind, _, args, _ in spy.calls if kind == "discrete_obstacles_terrain"]
        self.assertEqual(set(obstacles), {0.025, 0.15})
        np.testing.assert_allclose(terrain.env_origins[1, 19, :2], [12.0, 117.0])
        self.assertGreater(terrain.env_origins[1, 19, 2], 0)

    def test_selected_and_randomized_modes_and_cap_validation(self):
        cfg = config()
        cfg.mesh_type, cfg.num_rows, cfg.num_cols, cfg.border_size = "heightfield", 2, 2, 0.0
        cfg.selected, cfg.selected_type, cfg.selected_difficulty = True, "stairs_up", 1.0
        spy = PrimitiveSpy()
        with patch.dict(NS, terrain_utils=spy):
            terrain = NS["Q1Terrain"](cfg, 8)
        self.assertTrue((terrain.terrain_classes == 3).all())
        self.assertTrue(all(call[3]["step_height"] == 0.20 for call in spy.calls))
        cfg.curriculum, cfg.selected, cfg.num_cols = False, False, 20
        spy = PrimitiveSpy()
        with patch.dict(NS, terrain_utils=spy):
            terrain = NS["Q1Terrain"](cfg, 8)
        self.assertEqual(set(terrain.terrain_classes.flatten()), {-1, 0, 1, 2, 3, 4})
        cfg.stair_height_range = [0.025, 0.201]
        with self.assertRaisesRegex(ValueError, "stair risers"):
            NS["Q1Terrain"](cfg, 8)

    def test_simulation_uses_q1_generator_for_mesh_and_plane(self):
        cfg, spy = config(), PrimitiveSpy()
        cfg.num_rows, cfg.num_cols, cfg.border_size = 2, 2, 0.0
        with patch.dict(NS, terrain_utils=spy):
            terrain = NS["Q1Terrain"](cfg, 4)
        self.assertEqual(spy.mesh_call, ((160, 160), 0.1, 0.005, 0.75))
        self.assertEqual(terrain.triangles.shape, (1, 3))
        env = rough_env()
        env.cfg.terrain.mesh_type = "trimesh"
        env.sim_device_id = env.graphics_device_id = 0
        env.physics_engine = object()
        calls = []
        env.gym.create_sim = lambda *a: "simulation"
        env._create_trimesh = lambda: calls.append("mesh")
        env._create_envs = lambda: calls.append("robots")
        generator = lambda cfg, n: calls.append(("terrain", n)) or object()
        with patch.dict(NS, Q1Terrain=generator):
            env.create_sim()
        self.assertEqual(calls, [("terrain", 4), "mesh", "robots"])
        calls.clear()
        env.cfg.terrain.mesh_type = "heightfield"
        env._create_heightfield = lambda: calls.append("heightfield")
        with patch.dict(NS, Q1Terrain=generator):
            env.create_sim()
        self.assertEqual(calls, [("terrain", 4), "heightfield", "robots"])
        calls.clear()
        env.cfg.terrain.mesh_type = "plane"
        env._create_ground_plane = lambda: calls.append("plane")
        env.create_sim()
        self.assertEqual(calls, ["plane", "robots"])

    def test_height_normal_subsets_and_yaw_on_analytic_ramp(self):
        env = rough_env(slope=0.2, height=0.30)
        points = torch.tensor([[[4.03, 2.72], [4.06, 3.18]], [[6.11, 2.75], [5.16, 3.26]]])
        height, normal = env._terrain_surface(points)
        torch.testing.assert_close(height, points[..., 0]*0.2 + 0.3, atol=2e-6, rtol=0)
        expected = torch.tensor([-0.2, 0.0, 1.0])
        expected /= expected.norm()
        torch.testing.assert_close(normal, expected.expand_as(normal), atol=2e-6, rtol=0)
        env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
        heights = env._get_heights(torch.tensor([1, 3]))
        expected_height = 1.1 - 0.2*env.height_points[[1, 3], :, 1]
        torch.testing.assert_close(heights, expected_height, atol=2e-6, rtol=0)
        self.assertEqual(env._get_heights(torch.tensor([], dtype=torch.long)).shape, (0, 77))
        env._post_physics_step_callback()
        torch.testing.assert_close(env.base_height, torch.full((4,), 0.402), atol=1e-4, rtol=0)
        env.compute_observations()
        self.assertEqual(env.privileged_obs_buf.shape, (4, 151))
        self.assertTrue(torch.isfinite(env.privileged_obs_buf).all())

    def test_wheel_clearance_invariant_to_elevated_terrain(self):
        ground = rough_env(height=0.0)
        raised = rough_env(height=1.2)
        for height in (-1.2, 1.2):
            translated = rough_env(height=height)
            for a, b in zip(ground._sample_wheel_state(), translated._sample_wheel_state()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        raised.rigid_body_states[:, raised.feet_indices, 2] += 0.10
        clearance, _, touch, _ = raised._sample_wheel_state()
        self.assertTrue((clearance > 0.099).all())
        # Real support force wins over a sampled stair-edge clearance error.
        self.assertTrue((touch == 1).all())
        raised.contact_forces[:, raised.feet_indices] = 0
        self.assertEqual(raised._sample_wheel_state()[2].sum(), 0)

    def test_slope_tangential_slip_uses_local_contact_normal(self):
        env = rough_env(slope=0.2)
        _, normal = env._terrain_surface(env.rigid_body_states[:, env.feet_indices, :2])
        radius = env.cfg.asset.wheel_radius
        tangent = torch.tensor([1.0, 0.0, 0.2])
        tangent /= tangent.norm()
        wheel = env.rigid_body_states[:, env.feet_indices].clone()
        wheel[..., 7:10] = tangent*0.3
        wheel[..., 11] = 0.3/radius
        env.rigid_body_states[:, env.feet_indices] = wheel
        self.assertLess(env._sample_wheel_state()[3].abs().max(), 1e-6)
        wheel[..., 11] = 0
        env.rigid_body_states[:, env.feet_indices] = wheel
        torch.testing.assert_close(env._sample_wheel_state()[3], torch.full((4, 2), 0.3), atol=1e-6, rtol=0)

    def test_steps_stay_discrete_and_samples_clamp_at_map_border(self):
        env = rough_env()
        env.height_samples[41:, :] = 40  # independent 20 cm step at x=4.1
        height, normal = env._terrain_surface(torch.tensor([[4.04, 3.0], [4.14, 3.0]]))
        torch.testing.assert_close(height, torch.tensor([0.0, 0.20]))
        torch.testing.assert_close(normal, torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
        edge, _ = env._terrain_surface(torch.tensor([[-100., -100.], [10000., 10000.]]))
        self.assertTrue(torch.isfinite(edge).all())

    def test_flat_rewards_preserved_rough_legs_and_single_support_allowed(self):
        env = rough_env()
        env.env_class = torch.tensor([-1, 2, 3, 4])
        env.dof_pos[:, env.leg_dof_indices[0]] += 0.35
        env.dof_vel[:, env.leg_dof_indices[0]] = 1.0
        env._update_leg_kinematics()
        for method in (env._reward_actual_posture, env._reward_leg_symmetry,
                       env._reward_rolling_leg_motion, env._reward_nominal_state):
            cost = method()
            self.assertGreater(cost[0], 0)
            self.assertEqual(cost[1:].sum(), 0)
        env.episode_length_buf[:] = 100
        env.wheel_duty[:] = torch.tensor([1., 0.])
        self.assertGreater(env._reward_rolling_contact()[0], 0)
        self.assertEqual(env._reward_rolling_contact()[1:].sum(), 0)
        env.wheel_duty[:] = 0
        self.assertTrue((env._reward_rolling_contact() > 0).all())
        env.base_height[:] = 0.497
        self.assertGreater(env._reward_base_height()[1], env._reward_base_height()[0])

    def test_rough_commands_do_not_park_or_reverse_before_reaching_obstacle(self):
        torch.manual_seed(21)
        env = rough_env(2048)
        env.env_class[:1024] = -1
        env.path_active[:] = False
        env.needs_command_sample[:] = True
        ids = torch.arange(env.num_envs)
        env._resample_commands(ids)
        self.assertTrue((env.command_targets[1024:, 0].abs() >= 0.15).all())
        self.assertTrue((env.command_targets[:1024, 0] == 0).any())
        saved = env.command_targets[1024:, 0].clone()
        env.path_active[:] = True
        env.episode_length_buf[:] = int(4.0/env.dt)
        env._resample_commands(ids)
        torch.testing.assert_close(saved, env.command_targets[1024:, 0])
        env.cfg.commands.ranges.lin_vel_x = [0., 0.]
        env.path_active[:] = False
        env.needs_command_sample[:] = True
        env._resample_commands(ids)
        self.assertEqual(env.command_targets[:, 0].abs().sum(), 0)  # balance stage
        NS["configure_q1_stage"](env.cfg, "turning")
        env.needs_command_sample[:] = True
        env._resample_commands(ids)
        saved = env.command_targets[1024:, :2].clone()
        self.assertTrue((saved[:, 1] != 0).any())
        self.assertTrue(((saved[:, 0] == 0) & (saved[:, 1] != 0)).any())
        env.path_active[:] = False  # turning deliberately releases straight reference
        env.episode_length_buf += int(4.0/env.dt)
        env._resample_commands(ids)
        torch.testing.assert_close(saved, env.command_targets[1024:, :2])

    def test_origins_metadata_and_real_progress_curriculum(self):
        torch.manual_seed(4)
        env = rough_env()
        env._get_env_origins()
        self.assertEqual(env.env_class.tolist(), [-1, -1, 2, 2])
        self.assertEqual(env.terrain_y_max, 12.0)
        torch.testing.assert_close(env.env_origins,
                                   env.terrain_origins[env.terrain_levels, env.terrain_types])
        env.terrain_levels[:] = 1
        env.max_terrain_level = 3
        env.metric_steps[:] = 1500
        env.metric_sum[:, 0] = 0.02*1500
        env.terrain_command_distance[:] = 3.0
        env.terrain_travel_distance[:] = 2.8
        env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
        env.root_states[:, :2] = env.env_origins[:, :2]
        env.root_states[0, 0] += 2.5  # sufficient actual progress
        env.physical_failure = torch.tensor([False, True, False, False])
        env.time_out_buf[:] = True
        env.terrain_command_distance[3] = 0  # pure balance must not advance
        env._update_terrain_curriculum(torch.arange(4))
        self.assertEqual(env.terrain_levels.tolist(), [2, 0, 1, 1])
        # Repeated success remains at the top row; no out-of-bounds origins.
        env.root_states[0, 0] = env.env_origins[0, 0] + 2.5
        env._update_terrain_curriculum(torch.tensor([0]))
        self.assertEqual(env.terrain_levels[0], 2)
        env.cfg.terrain.curriculum = True
        env.reset_idx(torch.tensor([0, 2]))
        self.assertEqual(env.terrain_command_distance[[0, 2]].sum(), 0)
        self.assertEqual(env.terrain_travel_distance[[0, 2]].sum(), 0)
        torch.testing.assert_close(env.root_states[[0, 2], 2], env.env_origins[[0, 2], 2] + 0.402)
        torch.testing.assert_close(env.measured_heights[[0, 2]], env._get_heights(torch.tensor([0, 2])))

    def test_rough_playback_segment_duration_and_saved_schedule(self):
        env = rough_env(2)
        env.dt = 0.5
        train_cfg = NS["Q1CfgPPO"]()
        runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=object()),
                                 get_inference_policy=lambda device: lambda obs, hist: (torch.zeros(2, 6), None))
        registry = SimpleNamespace(get_cfgs=lambda task: (env.cfg, train_cfg),
                                   make_env=lambda *a, **kw: (env, env.cfg),
                                   make_alg_runner=lambda *a, **kw: (runner, train_cfg))
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            namespace = dict(NS, csv=csv, json=json, datetime=datetime, os=os,
                             WHEEL_LEGGED_GYM_ROOT_DIR=folder, task_registry=registry,
                             export_policy_as_jit=lambda *a: None)
            load_code("wheel_legged_gym/envs/q1/play_q1.py", namespace)
            namespace["play_q1"](SimpleNamespace(num_envs=2, q1_stage="low_speed", load_run="test", checkpoint=2,
                                                 headless=True, play_mode="auto", play_duration=60.5))
            output = Path(folder)/"logs/q1/exported/evaluations"
            summary = json.loads(next(output.glob("*.json")).read_text())
            self.assertEqual(summary["segment_s"], 20.0)
            self.assertEqual(summary["terrain_mesh"], "heightfield")
            with next(output.glob("*.csv")).open() as stream:
                rows = list(csv.reader(stream))[1:][::40]
            np.testing.assert_allclose([float(row[3]) for row in rows], [0.0, 0.2, 0.0, -0.2], atol=1e-7)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
