"""Exercise explicit playback terrain generation and the shared keyboard loop."""
import ast
from copy import deepcopy
from contextlib import redirect_stdout
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_q1_flat import NS, load_code, source
from test_q1_terrain import PrimitiveSpy
from test_q1_keyboard import Q1Keyboard, viewer_env, event


namespace = dict(NS, deepcopy=deepcopy, math=math)
load_code("wheel_legged_gym/envs/q1/keyboard_play.py", namespace)
configure = namespace["configure_keyboard_play"]


def args(**kwargs):
    values = dict(task="q1", headless=False, play_mode="keyboard", num_envs=None,
                  terrain="mixed", difficulty=0.5)
    values.update(kwargs)
    return SimpleNamespace(**values)


class KeyboardPlayTests(unittest.TestCase):
    def test_default_gallery_is_copy_and_keeps_training_configuration(self):
        original = NS["Q1Cfg"]()
        before = NS["class_to_dict"](original)
        with redirect_stdout(io.StringIO()) as output:
            cfg = configure(original, args())
        self.assertEqual(NS["class_to_dict"](original), before)
        self.assertEqual(cfg.env.num_envs, 6)
        self.assertEqual(cfg.terrain.playback_layout,
                         ["stairs_down", "stairs_up", "slope_up", "slope_down", "rough_slope", "flat"])
        self.assertIn("outward: UP stairs", output.getvalue())
        self.assertFalse(cfg.terrain.curriculum)
        self.assertFalse(cfg.noise.add_noise)
        self.assertFalse(cfg.domain_rand.push_robots)
        self.assertEqual((cfg.terrain.num_rows, cfg.terrain.num_cols), (1, 6))
        self.assertEqual(cfg.terrain.platform_size, original.terrain.platform_size)

    def test_small_counts_always_include_stairs_and_column_mapping_is_exact(self):
        for count in (1, 2, 6, 12):
            cfg = configure(NS["Q1Cfg"](), args(num_envs=count))
            cfg.terrain.mesh_type = "heightfield"
            cfg.terrain.border_size = 0
            spy = PrimitiveSpy()
            with patch.dict(NS, terrain_utils=spy):
                terrain = NS["Q1Terrain"](cfg.terrain, count)
            columns = torch.div(torch.arange(count), count/cfg.terrain.num_cols,
                                rounding_mode="floor").long()
            self.assertEqual(columns.tolist(), list(range(count)))
            expected = [cfg.terrain.terrain_labels[cfg.terrain.terrain_names.index(name)]
                        for name in cfg.terrain.playback_layout]
            self.assertEqual(terrain.terrain_classes[0].tolist(), expected)
            self.assertEqual(cfg.terrain.playback_layout[0], "stairs_down")
            stairs = [kw["step_height"] for kind, _, _, kw in spy.calls
                      if kind == "pyramid_stairs_terrain"]
            self.assertAlmostEqual(stairs[0], -0.1125)

    def test_selected_stairs_maximum_is_fixed_after_reconstruction(self):
        cfg = configure(NS["Q1Cfg"](), args(num_envs=2, terrain="stairs_up", difficulty=1))
        cfg.terrain.mesh_type, cfg.terrain.border_size = "heightfield", 0
        spy = PrimitiveSpy()
        with patch.dict(NS, terrain_utils=spy):
            first = NS["Q1Terrain"](cfg.terrain, 2)
        with patch.dict(NS, terrain_utils=PrimitiveSpy()):
            second = NS["Q1Terrain"](cfg.terrain, 2)
        stairs = [kw["step_height"] for kind, _, _, kw in spy.calls
                  if kind == "pyramid_stairs_terrain"]
        self.assertEqual(stairs, [0.2]*2)
        np.testing.assert_equal(first.env_origins, second.env_origins)

    def test_invalid_cli_values_fail_before_environment_creation(self):
        for override in (dict(num_envs=0), dict(num_envs=-1), dict(difficulty=-0.1),
                         dict(difficulty=1.1), dict(difficulty=float("nan")),
                         dict(headless=True), dict(play_mode="auto"), dict(task="q1_flat"),
                         dict(terrain="bogus")):
            with self.assertRaises(ValueError):
                configure(NS["Q1Cfg"](), args(**override))

    def test_camera_orbit_persists_and_follows_new_position(self):
        env = viewer_env(2)
        env.terrain_types = torch.tensor([0, 1])
        env.cfg.terrain.playback_layout = ["stairs_down", "stairs_up"]
        env.render = lambda: None
        p = SimpleNamespace(x=3.0, y=4.0, z=5.0)
        env.gym.get_viewer_camera_transform = lambda *unused: SimpleNamespace(p=p)
        controls = Q1Keyboard(env)
        controls.render()
        env.root_states[0, 0] += 0.5
        controls.render()
        self.assertEqual(env.cameras[-1][0], [3.5, 4.0, 5.0])
        self.assertEqual(controls.terrain_name(), "stairs_down")
        controls.handle_event(event("q1_select_1"))
        self.assertEqual(controls.terrain_name(), "stairs_up")

    def test_entry_defaults_and_existing_checkpoint_options_are_forwarded(self):
        calls = []
        def get_args(**kwargs):
            calls.append(kwargs)
            return args(load_run="trained_run", checkpoint=30000)
        entry = dict(__name__="test_entry", get_args=get_args,
                     GALLERY=namespace["GALLERY"], keyboard_play=calls.append)
        load_code("wheel_legged_gym/scripts/keyboard_play.py", entry)
        entry["main"]()
        self.assertEqual(calls[0]["default_task"], "q1")
        self.assertEqual(calls[1].checkpoint, 30000)
        self.assertEqual(calls[1].load_run, "trained_run")
        # Test the actual shared argument parser's default override too.
        tree = ast.parse(source("wheel_legged_gym/utils/helpers.py").read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args"]
        options = []
        def parse_arguments(**kwargs):
            options.extend(kwargs["custom_parameters"])
            return SimpleNamespace(compute_device_id=0, sim_device_type="cuda")
        parser = dict(gymutil=SimpleNamespace(parse_arguments=parse_arguments))
        exec(compile(tree, "helpers.py", "exec"), parser)
        parser["get_args"](default_task="q1")
        self.assertEqual(next(p["default"] for p in options if p["name"] == "--task"), "q1")

    def test_selecting_robot_during_render_does_not_corrupt_camera_offset(self):
        env = viewer_env(2)
        env.root_states[1, 1] = 8
        env.gym.get_viewer_camera_transform = lambda *unused: SimpleNamespace(
            p=SimpleNamespace(x=0, y=2, z=1.402))
        dispatched = []
        def render():
            if not dispatched:
                controls.handle_event(event("q1_select_1"))
                dispatched.append(True)
        env.render = render
        controls = Q1Keyboard(env)
        controls.render()
        self.assertEqual(controls.camera_offset, [0, 2, 1])
        controls.render()
        self.assertEqual(env.cameras[-1][0][1], 10)


if __name__ == "__main__":
    unittest.main()
