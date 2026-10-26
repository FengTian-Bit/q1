"""Directional +X climbing terrain with explicit spawn and goal metadata.

This module creates the actual integer height array directly. It deliberately
does not use pyramid stairs: the spawn is before a rising staircase, rather
than at the top of a hill. Metadata describes the quantized height array.
"""

import numpy as np
from isaacgym import terrain_utils

from wheel_legged_gym.utils.terrain import Terrain


class CTBCTerrain(Terrain):
    def __init__(self, cfg, num_robots):
        if cfg.mesh_type not in ("heightfield", "trimesh"):
            raise ValueError("CTBC terrain requires a heightfield or trimesh")
        if cfg.num_rows < 1 or cfg.num_cols < 1:
            raise ValueError("CTBC terrain needs positive row and column counts")
        weights = np.asarray(cfg.terrain_proportions, dtype=float)
        if (len(weights) != 2 or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0)
                or list(cfg.terrain_names) != ["flat", "stairs_up"]):
            raise ValueError("CTBC terrain requires normalized flat/stairs_up proportions")
        if cfg.horizontal_scale <= 0 or cfg.vertical_scale <= 0:
            raise ValueError("CTBC terrain scales must be positive")
        shape = (cfg.num_rows, cfg.num_cols)
        self.terrain_classes = np.full(shape, -1, dtype=np.int64)
        self.riser_heights = np.zeros(shape)
        self.tread_widths = np.zeros(shape)
        self.step_counts = np.zeros(shape, dtype=np.int64)
        self.first_riser_x = np.zeros(shape)
        self.goal_x = np.zeros(shape)
        super().__init__(cfg, num_robots)

    def curiculum(self):
        if self.cfg.selected:
            self.selected_terrain()
            return
        for row in range(self.cfg.num_rows):
            difficulty = row / max(self.cfg.num_rows - 1, 1)
            for col in range(self.cfg.num_cols):
                self.add_terrain_to_map(self.make_terrain(
                    (col + 0.5) / self.cfg.num_cols, difficulty), row, col)

    def randomized_terrain(self):
        # Evaluation with curriculum off still has deterministic geometry and
        # coverage. Selected values provide a fixed-height evaluation sweep.
        self.curiculum()

    def selected_terrain(self):
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                self.add_terrain_to_map(self.make_terrain(0.0, 0.0), row, col)

    def make_terrain(self, choice, difficulty):
        cfg = self.cfg
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        lerp = lambda bounds: bounds[0] + difficulty * (bounds[1] - bounds[0])
        if cfg.selected:
            flat = cfg.selected_flat
            height, tread, steps = cfg.selected_height, cfg.selected_tread, cfg.selected_steps
        else:
            flat = choice < self.proportions[0]
            height, tread, steps = lerp(cfg.riser_height_range), lerp(cfg.tread_width_range), cfg.num_steps
        if (not np.isfinite(height) or not np.isfinite(tread)
                or height <= 0 or tread <= 0 or steps < 1 or int(steps) != steps):
            raise ValueError("CTBC riser height, tread and integer step count must be positive")
        horizontal, vertical = cfg.horizontal_scale, cfg.vertical_scale
        # The small tolerance prevents exact decimal grid values being rounded
        # down by binary floating-point representation.
        height_units = int(np.floor(height / vertical + 1e-9))
        tread_cells = int(np.floor(tread / horizontal + 1e-9))
        first_cell = int(np.floor(cfg.first_step_x / horizontal + 1e-9))
        if height_units < 1 or tread_cells < 1:
            raise ValueError("CTBC riser height and tread must cover at least one grid cell")
        first_x = first_cell * horizontal
        last_x = first_x + (int(steps) - 1) * tread_cells * horizontal
        goal = cfg.spawn_x + cfg.flat_goal_distance if flat else last_x + cfg.goal_margin
        if (not 0 < cfg.spawn_x < first_x or cfg.goal_margin <= 0
                or not cfg.spawn_x < goal < cfg.terrain_length
                or last_x >= cfg.terrain_length):
            raise ValueError("CTBC spawn, steps and goal must fit in the terrain tile")
        if height_units * int(steps) > np.iinfo(np.int16).max:
            raise ValueError("CTBC staircase exceeds int16 heightfield range")
        terrain = terrain_utils.SubTerrain(
            "ctbc_flat" if flat else "ctbc_stairs_up",
            width=self.length_per_env_pixels, length=self.width_per_env_pixels,
            vertical_scale=vertical, horizontal_scale=horizontal)
        terrain.terrain_type = -1 if flat else 3
        terrain.riser_height = 0.0 if flat else height_units * vertical
        terrain.tread_width = tread_cells * horizontal
        terrain.step_count = 0 if flat else int(steps)
        terrain.first_riser_x = first_x
        terrain.goal_x = goal
        if not flat:
            for step in range(int(steps)):
                terrain.height_field_raw[first_cell + step * tread_cells:, :] += height_units
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        # Copy without Terrain.add_terrain_to_map's centre/max-height origin:
        # the centre can lie on the upper platform, while CTBC spawns at z=0.
        start_x = self.border + row * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        self.height_field_raw[
            start_x:start_x + self.length_per_env_pixels,
            start_y:start_y + self.width_per_env_pixels] = terrain.height_field_raw
        offset_x = row * self.env_length
        self.env_origins[row, col] = [offset_x + self.cfg.spawn_x,
                                      (col + 0.5) * self.env_width, 0.0]
        self.terrain_classes[row, col] = terrain.terrain_type
        self.riser_heights[row, col] = terrain.riser_height
        self.tread_widths[row, col] = terrain.tread_width
        self.step_counts[row, col] = terrain.step_count
        self.first_riser_x[row, col] = offset_x + terrain.first_riser_x
        self.goal_x[row, col] = offset_x + terrain.goal_x
