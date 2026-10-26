"""Inspect CTBC's real height arrays and stage isolation, without PhysX."""
from types import SimpleNamespace
import unittest

import numpy as np

from test_q1_flat import NS, load_code


class HeightfieldTools:
    def SubTerrain(self, name, width, length, vertical_scale, horizontal_scale):
        return SimpleNamespace(
            name=name, width=width, length=length,
            vertical_scale=vertical_scale, horizontal_scale=horizontal_scale,
            height_field_raw=np.zeros((width, length), dtype=np.int16))

    def convert_heightfield_to_trimesh(self, heights, horizontal_scale,
                                     vertical_scale, slope_threshold):
        # Geometry tests below inspect the actual input height array. This
        # return value only permits testing the mesh-construction code path.
        return np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32)


namespace = dict(NS, terrain_utils=HeightfieldTools())
load_code("wheel_legged_gym/envs/q1_ctbc/q1_ctbc_config.py", namespace)
load_code("wheel_legged_gym/utils/terrain.py", namespace)
load_code("wheel_legged_gym/envs/q1_ctbc/terrain.py", namespace)
Config, PPOConfig = namespace["Q1CTBCCfg"], namespace["Q1CTBCCfgPPO"]
Terrain = namespace["CTBCTerrain"]
configure = namespace["configure_ctbc_stage"]


def make_terrain(stage="single", **overrides):
    cfg = configure(Config(), stage).terrain
    cfg.mesh_type = "heightfield"
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return Terrain(cfg, 32)


def tile(terrain, row, col):
    start_x = terrain.border + row * terrain.length_per_env_pixels
    start_y = terrain.border + col * terrain.width_per_env_pixels
    return terrain.height_field_raw[
        start_x:start_x + terrain.length_per_env_pixels,
        start_y:start_y + terrain.width_per_env_pixels]


class CTBCTerrainTests(unittest.TestCase):
    def test_default_single_step_points_up_from_spawn_and_has_exact_goals(self):
        terrain = make_terrain()
        self.assertEqual((terrain.length_per_env_pixels, terrain.width_per_env_pixels), (240, 120))
        self.assertEqual(terrain.height_field_raw.shape, (1600, 760))
        self.assertEqual(terrain.terrain_classes[0].tolist(), [-1, 3, 3, 3, 3])
        for row in range(6):
            for col in range(5):
                heights = tile(terrain, row, col)
                np.testing.assert_allclose(terrain.env_origins[row, col],
                                           [row * 6.0 + 1.0, (col + 0.5) * 3.0, 0.0])
                self.assertTrue((heights[:68] == 0).all())
                self.assertTrue((np.diff(heights.astype(np.int32), axis=0) >= 0).all())
                self.assertAlmostEqual(terrain.first_riser_x[row, col], row * 6.0 + 1.7)
                if col == 0:
                    self.assertEqual(terrain.step_counts[row, col], 0)
                    self.assertTrue((heights == 0).all())
                    self.assertAlmostEqual(terrain.goal_x[row, col], row * 6.0 + 2.8)
                else:
                    self.assertEqual(terrain.step_counts[row, col], 1)
                    np.testing.assert_allclose(heights[68:] * 0.005,
                                               terrain.riser_heights[row, col])
                    self.assertAlmostEqual(terrain.goal_x[row, col], row * 6.0 + 2.0)
        self.assertAlmostEqual(terrain.riser_heights[0, 1], 0.025)
        self.assertAlmostEqual(terrain.riser_heights[-1, 1], 0.08)

    def test_three_steps_have_measured_risers_and_treads_not_pyramid_geometry(self):
        terrain = make_terrain("stairs")
        for row in range(6):
            heights = tile(terrain, row, 1)[:, 60]
            edges = np.flatnonzero(np.diff(heights.astype(np.int32))) + 1
            self.assertEqual(len(edges), 3)
            self.assertEqual(edges[0], 68)
            np.testing.assert_allclose(np.diff(edges) * 0.025, terrain.tread_widths[row, 1])
            np.testing.assert_allclose((heights[edges] - heights[edges - 1]) * 0.005,
                                       terrain.riser_heights[row, 1])
            self.assertAlmostEqual(terrain.goal_x[row, 1], row * 6.0 + edges[-1] * 0.025 + 0.3)
            self.assertEqual(heights[40], 0)
            self.assertAlmostEqual(heights[-1] * 0.005, 3 * terrain.riser_heights[row, 1])
        self.assertAlmostEqual(terrain.riser_heights[0, 1], 0.04)
        self.assertAlmostEqual(terrain.riser_heights[-1, 1], 0.12)
        self.assertAlmostEqual(terrain.tread_widths[0, 1], 0.5)
        self.assertAlmostEqual(terrain.tread_widths[-1, 1], 0.35)

    def test_selected_evaluation_is_quantized_and_repeatable_without_curriculum(self):
        overrides = dict(curriculum=False, selected=True, selected_height=0.073,
                         selected_tread=0.483, selected_steps=3)
        first, second = make_terrain(**overrides), make_terrain(**overrides)
        np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
        np.testing.assert_allclose(first.riser_heights, 0.07)
        np.testing.assert_allclose(first.tread_widths, 0.475)
        self.assertTrue((first.step_counts == 3).all())
        self.assertTrue((first.terrain_classes == 3).all())
        self.assertTrue((first.env_origins[:, :, 2] == 0).all())
        flat = make_terrain(**dict(overrides, selected_flat=True))
        self.assertTrue((flat.height_field_raw == 0).all())
        self.assertTrue((flat.terrain_classes == -1).all())
        self.assertTrue((flat.step_counts == 0).all())

    def test_curriculum_off_keeps_same_default_geometry_and_mesh_path(self):
        first = make_terrain()
        second = make_terrain(curriculum=False)
        np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
        cfg = Config().terrain
        terrain = Terrain(cfg, 32)
        self.assertEqual(terrain.vertices.shape, (3, 3))
        np.testing.assert_array_equal(first.height_field_raw, terrain.height_field_raw)

    def test_invalid_geometry_is_rejected_before_silent_empty_steps(self):
        for overrides in (dict(selected_height=0), dict(selected_height=0.001),
                          dict(selected_tread=0.01), dict(selected_steps=0),
                          dict(selected_steps=1.5), dict(selected_steps=20),
                          dict(spawn_x=2.0), dict(first_step_x=7.0),
                          dict(goal_margin=0), dict(goal_margin=8.0)):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_terrain(selected=True, **overrides)

    def test_stages_reset_robust_flags_and_do_not_mutate_q1_or_another_instance(self):
        original = namespace["Q1Cfg"]()
        before = namespace["class_to_dict"](original)
        untouched = Config()
        cfg = configure(Config(), "robust")
        self.assertTrue(cfg.domain_rand.randomize_friction)
        self.assertTrue(cfg.domain_rand.randomize_action_delay)
        self.assertTrue(cfg.noise.add_noise)
        self.assertEqual(cfg.terrain.num_steps, 3)
        configure(cfg, "single")
        self.assertFalse(cfg.domain_rand.randomize_friction)
        self.assertFalse(cfg.domain_rand.randomize_action_delay)
        self.assertFalse(cfg.noise.add_noise)
        self.assertEqual(cfg.domain_rand.delay_ms_range, [0, 0])
        self.assertEqual(cfg.terrain.riser_height_range, [0.025, 0.08])
        self.assertEqual(cfg.terrain.num_steps, 1)
        self.assertEqual(untouched.terrain.num_steps, 1)
        self.assertEqual(untouched.domain_rand.friction_range, original.domain_rand.friction_range)
        self.assertEqual(namespace["class_to_dict"](original), before)
        with self.assertRaisesRegex(ValueError, "Unknown CTBC stage"):
            configure(cfg, "turning")

    def test_checkpoint_layout_and_specialized_config_are_explicit(self):
        cfg, train = Config(), PPOConfig()
        self.assertEqual((cfg.env.num_envs, cfg.env.num_observations,
                          cfg.env.num_privileged_obs, train.policy.num_encoder_obs),
                         (4096, 31, 151, 155))
        self.assertEqual(cfg.control.leg_action_scales, [0.5, 0.9, 0.5, 0.9])
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [0.15, 0.3])
        self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [0.0, 0.0])
        self.assertEqual(train.runner.max_iterations, 30000)
        self.assertFalse(train.runner.periodic_video.enabled)
        self.assertEqual(cfg.rewards.rough_leg_symmetry_weight, 0.0)
        self.assertEqual(cfg.rewards.scales.ctbc_same_x, 0.0)
        self.assertEqual((cfg.ctbc.contact_frames, cfg.ctbc.feedforward_hold_iterations,
                          cfg.ctbc.feedforward_anneal_iterations), (3, 3000, 12000))


if __name__ == "__main__":
    unittest.main()
