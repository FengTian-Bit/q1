"""Verify T0 terrain arguments, distributions, origins and stair quantization.

The stair double implements Gym's integer pyramid construction; it does not
simulate PhysX. Other primitive doubles record arguments and mark the surface.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TerrainTools:
    def __init__(self):
        self.calls = []

    def SubTerrain(self, name, width, length, vertical_scale, horizontal_scale):
        return SimpleNamespace(
            name=name, width=width, length=length,
            vertical_scale=vertical_scale, horizontal_scale=horizontal_scale,
            height_field_raw=np.zeros((width, length), dtype=np.int16))

    def pyramid_stairs_terrain(self, terrain, step_width, step_height, platform_size):
        self.calls.append(("stairs", terrain.name, dict(
            step_width=step_width, step_height=step_height, platform_size=platform_size)))
        stride = int(step_width / terrain.horizontal_scale)
        height = int(step_height / terrain.vertical_scale)
        platform = int(platform_size / terrain.horizontal_scale)
        x0, x1, y0, y1, level = 0, terrain.width, 0, terrain.length, height
        while x1-x0 > platform and y1-y0 > platform:
            x0, x1, y0, y1 = x0+stride, x1-stride, y0+stride, y1-stride
            terrain.height_field_raw[x0:x1, y0:y1] = level
            level += height

    def pyramid_sloped_terrain(self, terrain, **kwargs):
        self.calls.append(("slope", terrain.name, kwargs))
        terrain.height_field_raw[:] = 1

    def random_uniform_terrain(self, terrain, **kwargs):
        self.calls.append(("roughness", terrain.name, kwargs))
        terrain.height_field_raw[:] += 1

    def discrete_obstacles_terrain(self, terrain, height, min_size, max_size, count, **kwargs):
        self.calls.append(("discrete", terrain.name, dict(
            height=height, min_size=min_size, max_size=max_size, count=count, **kwargs)))
        terrain.height_field_raw[:] = 3

    def convert_heightfield_to_trimesh(self, heights, horizontal_scale, vertical_scale, threshold):
        self.mesh_call = (heights.copy(), horizontal_scale, vertical_scale, threshold)
        return np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32)


def config(**overrides):
    cfg = SimpleNamespace(
        mesh_type="heightfield", curriculum=True, selected=False,
        horizontal_scale=0.1, vertical_scale=0.005, border_size=25.0,
        terrain_length=8.0, terrain_width=8.0, num_rows=20, num_cols=20,
        slope_treshold=0.75, difficulty_range=[0.0, 0.95], seed=17,
        terrain_names=["flat", "slope_down", "slope_up", "rough_slope",
                       "stairs_down", "stairs_up", "discrete"],
        terrain_labels=[-1, 0, 0, 1, 2, 3, 4],
        terrain_proportions=[0.1, 0.1, 0.1, 0.15, 0.2, 0.2, 0.15],
        slope_range=[0.1, 0.7], rough_slope_range=[0.1, 0.5],
        rough_height_range=[-0.05, 0.05], rough_height_step=0.005,
        rough_downsampled_scale=0.2, stair_height_range=[0.05, 0.25],
        stair_width=0.25, obstacle_height_range=[0.05, 0.30],
        obstacle_size_range=[1.0, 2.0], num_obstacles=20, platform_size=3.0,
        selected_type="stairs_down", selected_difficulty=0.95)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def build(cfg=None):
    tools = TerrainTools()
    namespace = dict(np=np, terrain_utils=tools, LeggedRobotCfg=SimpleNamespace(terrain=object))
    for relative in ("wheel_legged_gym/utils/terrain.py", "wheel_legged_gym/envs/q1/t0_terrain.py"):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Q1T0Terrain"](cfg or config(), 4096), tools


def tile(terrain, row, col):
    x0 = terrain.border + row * terrain.length_per_env_pixels
    y0 = terrain.border + col * terrain.width_per_env_pixels
    return terrain.height_field_raw[x0:x0+terrain.length_per_env_pixels,
                                    y0:y0+terrain.width_per_env_pixels]


class Q1T0TerrainTests(unittest.TestCase):
    def test_full_map_t0_column_distribution_and_endpoint_difficulty(self):
        terrain, _ = build()
        self.assertEqual(terrain.height_field_raw.shape, (2100, 2100))
        self.assertEqual(terrain.terrain_classes[0].tolist(),
                         [-1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
        np.testing.assert_allclose(terrain.difficulties[:, 0], np.linspace(0.0, 0.95, 20))
        self.assertAlmostEqual(terrain.riser_heights[0, 9], 0.05)
        self.assertAlmostEqual(terrain.riser_heights[-1, 9], 0.24)
        self.assertTrue((terrain.step_counts[:, :9] == 0).all())
        self.assertTrue((terrain.step_counts[:, 9:17] == 13).all())
        self.assertTrue((terrain.step_counts[:, 17:] == 0).all())
        np.testing.assert_allclose(terrain.env_origins[-1, -1, :2], [156.0, 156.0])
        self.assertTrue((terrain.height_field_raw[:250] == 0).all())

    def test_primitive_parameters_match_t0_with_unscaled_severity(self):
        _, tools = build(config(num_rows=2, border_size=0.0))
        slopes = [args["slope"] for kind, _, args in tools.calls if kind == "slope"]
        np.testing.assert_allclose(sorted(set(round(value, 9) for value in slopes)),
                                   [-0.67, -0.1, 0.1, 0.48, 0.67])
        rough = [args for kind, _, args in tools.calls if kind == "roughness"]
        self.assertTrue(all(args == dict(min_height=-0.05, max_height=0.05,
                                        step=0.005, downsampled_scale=0.2) for args in rough))
        stairs = [args for kind, _, args in tools.calls if kind == "stairs"]
        self.assertEqual(set(round(args["step_height"], 8) for args in stairs), {-0.24, -0.05, 0.05, 0.24})
        self.assertTrue(all(args["step_width"] == 0.25 and args["platform_size"] == 3.0 for args in stairs))
        discrete = [args for kind, _, args in tools.calls if kind == "discrete"]
        self.assertEqual(set(args["height"] for args in discrete), {0.05, 0.2875})
        self.assertTrue(all(args["min_size"] == 1 and args["max_size"] == 2
                            and args["count"] == 20 for args in discrete))

    def test_centre_spawn_stair_signs_quantization_and_sideways_faces(self):
        terrain, _ = build(config(num_rows=2, border_size=0.0))
        for col, sign in ((9, -1), (13, 1)):
            heights = tile(terrain, 1, col).astype(np.int32)
            self.assertEqual(heights[40, 40], sign * 13 * 48)
            self.assertEqual(heights[0, 40], 0)
            self.assertEqual(heights[40, 0], 0)
            # Gym truncates nominal 25 cm treads to two 10 cm cells.
            self.assertAlmostEqual(terrain.tread_widths[1, col], 0.20)
            edge_indices = np.flatnonzero(np.diff(heights[:41, 40])) + 1
            np.testing.assert_array_equal(edge_indices, np.arange(2, 27, 2))
            np.testing.assert_array_equal(heights[:, 40], heights[40, :])
            self.assertAlmostEqual(terrain.env_origins[1, col, 2], sign * 3.12)
            # At a centre origin, outward motion on positive stairs goes down.
            self.assertEqual(np.sign(heights[0, 40]-heights[40, 40]), -sign)

    def test_rectangular_tiles_preserve_x_y_shape_and_centre_origins(self):
        terrain, _ = build(config(num_rows=2, num_cols=2, terrain_width=6.0,
                                  border_size=0.0, selected=True, selected_type="stairs_down"))
        self.assertEqual(terrain.height_field_raw.shape, (160, 120))
        np.testing.assert_allclose(terrain.env_origins[1, 1, :2], [12.0, 9.0])
        self.assertEqual(terrain.step_counts[0, 0], 8)
        self.assertEqual(tile(terrain, 0, 0).shape, (80, 60))
        self.assertAlmostEqual(terrain.env_origins[0, 0, 2], -8 * 0.24)

    def test_random_mode_matches_t0_independent_type_and_three_difficulty_draws(self):
        cfg = config(curriculum=False, num_rows=8, num_cols=20, border_size=0.0)
        terrain, _ = build(cfg)
        expected_rng = np.random.default_rng(cfg.seed)
        expected_labels = np.empty((8, 20), dtype=np.int64)
        expected_difficulty = np.empty((8, 20))
        cumulative = np.cumsum(cfg.terrain_proportions)
        for col in range(20):
            for row in range(8):
                expected_difficulty[row, col] = 0.95 * expected_rng.choice([0.5, 0.75, 0.9])
                index = np.searchsorted(cumulative, expected_rng.uniform(), side="left")
                expected_labels[row, col] = cfg.terrain_labels[index]
        np.testing.assert_array_equal(terrain.terrain_classes, expected_labels)
        np.testing.assert_allclose(terrain.difficulties, expected_difficulty)
        self.assertEqual(set(np.round(terrain.difficulties.flatten(), 8)), {0.475, 0.7125, 0.855})
        repeat, _ = build(deepcopy(cfg))
        np.testing.assert_array_equal(terrain.height_field_raw, repeat.height_field_raw)

    def test_selected_geometry_overrides_curriculum_without_changing_difficulty(self):
        for curriculum in (True, False):
            terrain, _ = build(config(selected=True, curriculum=curriculum,
                                      selected_type="stairs_up", selected_difficulty=1.0,
                                      num_rows=2, num_cols=2, border_size=0.0))
            self.assertTrue((terrain.terrain_classes == 3).all())
            self.assertTrue((terrain.difficulties == 1.0).all())
            np.testing.assert_allclose(terrain.riser_heights, 0.25)
            self.assertAlmostEqual(terrain.env_origins[0, 0, 2], 3.25)

    def test_explicit_playback_layout_and_mesh_conversion(self):
        terrain, tools = build(config(curriculum=False, num_rows=1, num_cols=3,
                                      mesh_type="trimesh", border_size=0.0,
                                      playback_layout=["stairs_down", "flat", "stairs_up"],
                                      playback_difficulty=0.5))
        np.testing.assert_array_equal(terrain.terrain_classes, [[2, -1, 3]])
        self.assertEqual(tools.mesh_call[1:], (0.1, 0.005, 0.75))
        np.testing.assert_array_equal(tools.mesh_call[0], terrain.height_field_raw)
        self.assertEqual(terrain.triangles.shape, (1, 3))

    def test_invalid_configuration_fails_before_primitive_calls(self):
        cases = [dict(difficulty_range=[0.95, 0]), dict(terrain_proportions=[0.2]*7),
                 dict(terrain_labels=[0]*7), dict(horizontal_scale=0),
                 dict(selected=True, selected_type="highplatform"),
                 dict(selected=True, selected_difficulty=float("nan")),
                 dict(selected=True, stair_width=0.05)]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                build(config(**overrides))


if __name__ == "__main__":
    unittest.main(verbosity=2)
