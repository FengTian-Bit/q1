# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""T0's effective mixed-terrain distribution for the six-action Q1 task.

Geometry uses the same Isaac Gym primitives and their integer quantization as
Dreamwaq-highplatform's ordinary T0 task. Its disabled hollow/high-platform
generators are not needed here. Positive pyramid stairs rise towards the tile
centre: a centre-spawn robot drives *down* on ``stairs_up`` and *up* on the
inverted ``stairs_down`` tile, retaining T0's original naming convention.
"""

import numpy as np
from isaacgym import terrain_utils

from wheel_legged_gym.utils.terrain import Terrain


class Q1T0Terrain(Terrain):
    """Gym-compatible terrain with centre origins and per-tile teacher metadata."""

    TYPE_LABELS = {"flat": -1, "slope_down": 0, "slope_up": 0,
                   "rough_slope": 1, "stairs_down": 2, "stairs_up": 3,
                   "discrete": 4}

    def __init__(self, cfg, num_robots):
        if cfg.mesh_type not in ("heightfield", "trimesh"):
            raise ValueError("Q1 T0 terrain requires a heightfield or trimesh")
        if cfg.num_rows < 1 or cfg.num_cols < 1:
            raise ValueError("Q1 T0 terrain needs positive row and column counts")
        if (not np.isfinite(cfg.horizontal_scale) or cfg.horizontal_scale <= 0
                or not np.isfinite(cfg.vertical_scale) or cfg.vertical_scale <= 0
                or cfg.terrain_length <= 2.0 or cfg.terrain_width <= 2.0):
            raise ValueError("Q1 T0 terrain needs positive scales and tiles larger than 2 m")
        names, labels = list(cfg.terrain_names), list(cfg.terrain_labels)
        weights = np.asarray(cfg.terrain_proportions, dtype=float)
        if (len(names) != len(set(names)) or len(names) != len(labels)
                or len(names) != len(weights) or not len(names)
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0)
                or any(name not in self.TYPE_LABELS for name in names)
                or labels != [self.TYPE_LABELS[name] for name in names]):
            raise ValueError("Q1 T0 names, labels and normalized proportions must agree")
        bounds = np.asarray(getattr(cfg, "difficulty_range", [0.0, 0.95]), dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or not 0 <= bounds[0] <= bounds[1] <= 1):
            raise ValueError("Q1 T0 difficulty_range must be ordered within [0, 1]")
        self.difficulty_range = bounds
        self.rng = np.random.default_rng(getattr(cfg, "seed", None))
        shape = (cfg.num_rows, cfg.num_cols)
        self.terrain_classes = np.full(shape, -1, dtype=np.int64)
        self.riser_heights = np.zeros(shape)
        self.tread_widths = np.zeros(shape)
        self.step_counts = np.zeros(shape, dtype=np.int64)
        self.difficulties = np.zeros(shape)
        super().__init__(cfg, num_robots)

    def curiculum(self):
        if self.cfg.selected:
            self.selected_terrain()
            return
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = self._difficulty(row)
                choice = (col + 0.5) / self.cfg.num_cols
                self.add_terrain_to_map(self.make_terrain(choice, difficulty), row, col)

    def _difficulty(self, row):
        low, high = self.difficulty_range
        alpha = (row / max(1, self.cfg.num_rows - 1) if self.cfg.curriculum
                 else self.rng.choice([0.5, 0.75, 0.9]))
        return float(low + (high - low) * alpha)

    def randomized_terrain(self):
        layout = getattr(self.cfg, "playback_layout", None)
        if layout:
            if self.cfg.num_rows != 1 or len(layout) != self.cfg.num_cols:
                raise ValueError("T0 playback needs one explicit terrain column per robot")
            for col, name in enumerate(layout):
                self.add_terrain_to_map(self._make_named_terrain(
                    name, self.cfg.playback_difficulty), 0, col)
            return
        # Match T0's independent tile-type draw and its three-point difficulty
        # distribution when the curriculum is disabled.
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = self._difficulty(row)
                self.add_terrain_to_map(self.make_terrain(
                    self.rng.uniform(0.0, 1.0), difficulty), row, col)

    def selected_terrain(self):
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                self.add_terrain_to_map(self._make_named_terrain(
                    self.cfg.selected_type, self.cfg.selected_difficulty), row, col)

    def make_terrain(self, choice, difficulty):
        # T0 selects the first cumulative proportion >= the column cursor.
        index = min(int(np.searchsorted(self.proportions, choice, side="left")),
                    len(self.cfg.terrain_names) - 1)
        return self._make_named_terrain(self.cfg.terrain_names[index], difficulty)

    def _make_named_terrain(self, name, difficulty):
        cfg = self.cfg
        if name not in cfg.terrain_names or cfg.terrain_proportions[cfg.terrain_names.index(name)] <= 0:
            raise ValueError("Selected T0 terrain must have a positive configured proportion")
        if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise ValueError("T0 terrain difficulty must be within [0, 1]")
        lerp = lambda bounds: bounds[0] + (bounds[1] - bounds[0]) * difficulty
        terrain = terrain_utils.SubTerrain(
            "q1_t0_" + name, width=self.length_per_env_pixels,
            length=self.width_per_env_pixels, vertical_scale=cfg.vertical_scale,
            horizontal_scale=cfg.horizontal_scale)
        terrain.terrain_type = self.TYPE_LABELS[name]
        terrain.difficulty = difficulty
        terrain.riser_height, terrain.tread_width, terrain.step_count = 0.0, 0.0, 0
        if name in ("slope_down", "slope_up", "rough_slope"):
            slope = lerp(cfg.rough_slope_range if name == "rough_slope" else cfg.slope_range)
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=-slope if name == "slope_down" else slope,
                platform_size=cfg.platform_size)
            if name == "rough_slope":
                terrain_utils.random_uniform_terrain(
                    terrain, min_height=cfg.rough_height_range[0],
                    max_height=cfg.rough_height_range[1], step=cfg.rough_height_step,
                    downsampled_scale=cfg.rough_downsampled_scale)
        elif name in ("stairs_down", "stairs_up"):
            height = lerp(cfg.stair_height_range)
            width_cells = int(cfg.stair_width / cfg.horizontal_scale)
            height_units = int(height / cfg.vertical_scale)
            if width_cells < 1 or height_units < 1:
                raise ValueError("T0 stair width and height must cover at least one grid cell")
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=cfg.stair_width,
                step_height=-height if name == "stairs_down" else height,
                platform_size=cfg.platform_size)
            terrain.riser_height = height_units * cfg.vertical_scale
            terrain.tread_width = width_cells * cfg.horizontal_scale
            remaining_x, remaining_y = self.length_per_env_pixels, self.width_per_env_pixels
            platform_cells = int(cfg.platform_size / cfg.horizontal_scale)
            while remaining_x > platform_cells and remaining_y > platform_cells:
                terrain.step_count += 1
                remaining_x -= 2 * width_cells
                remaining_y -= 2 * width_cells
        elif name == "discrete":
            terrain_utils.discrete_obstacles_terrain(
                terrain, lerp(cfg.obstacle_height_range), cfg.obstacle_size_range[0],
                cfg.obstacle_size_range[1], cfg.num_obstacles, platform_size=cfg.platform_size)
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        # Keep the original centre/max-height origin used by the T0 generator.
        super().add_terrain_to_map(terrain, row, col)
        self.terrain_classes[row, col] = terrain.terrain_type
        self.riser_heights[row, col] = terrain.riser_height
        self.tread_widths[row, col] = terrain.tread_width
        self.step_counts[row, col] = terrain.step_count
        self.difficulties[row, col] = terrain.difficulty
