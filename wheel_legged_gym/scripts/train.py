# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
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

import numpy as np
import os
from datetime import datetime

import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry
import torch


def train(args):
    if args.task in ("q1_flat", "q1"):
        env_cfg, q1_train_cfg = task_registry.get_cfgs(args.task)
        if args.q1_stage == "balance":
            env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
            env_cfg.commands.stand_probability = 1.0
        if args.run_name is None:
            q1_train_cfg.runner.run_name = args.q1_stage
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    task_registry.save_cfgs(name=args.task)
    if args.task in ("q1_flat", "q1"):
        import json
        from shutil import copyfile
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.utils.helpers import class_to_dict
        with open(os.path.join(task_registry.log_dir, "resolved_config.json"), "w") as output:
            json.dump({"stage": args.q1_stage, "env": class_to_dict(env_cfg),
                       "train": class_to_dict(train_cfg)}, output, indent=2)
        asset_path = env_cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        copyfile(asset_path, os.path.join(task_registry.log_dir, "q1.urdf"))
        if args.task == "q1":
            for filename in ("q1_flat.py", "q1_flat_config.py"):
                copyfile(os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "wheel_legged_gym", "envs", "q1_flat", filename),
                         os.path.join(task_registry.log_dir, "parent_"+filename))
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
