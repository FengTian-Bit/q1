"""Deterministic Q1 stop/forward/reverse evaluation and pre-reset diagnostics."""
import csv
from datetime import datetime
import json
import os

import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils import export_policy_as_jit, task_registry
from .q1 import DIAGNOSTIC_FIELDS


def play_q1(args):
    cfg, train_cfg = task_registry.get_cfgs("q1")
    cfg.env.num_envs = 16 if args.num_envs is None else args.num_envs
    cfg.init_state.reset_lin_vel = 0.0
    cfg.init_state.reset_ang_vel = 0.0
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    cfg.commands.stand_probability = 1.0
    env, _ = task_registry.make_env("q1", args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    # get_load_path ignores exported/; do not create a spurious latest run.
    output = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported")
    export_policy_as_jit(runner.alg.actor_critic, os.path.join(output, "policies"))
    output = os.path.join(output, "evaluations")
    os.makedirs(output, exist_ok=True)
    stem = os.path.join(output, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    schedule = [0.0] if args.q1_stage == "balance" else [0.0, 0.2, 0.0, -0.2, 0.0, 0.3, 0.0, -0.3, 0.0]
    accumulated = torch.zeros(len(DIAGNOSTIC_FIELDS), device=env.device)
    count, resets = 0, 0
    with open(stem+".csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "env0_reset", "env0_failure"]+DIAGNOSTIC_FIELDS)
        for step in range(int(72.0/env.dt)):
            with torch.inference_mode():
                env.set_command_target(schedule[int(step*env.dt/4.0) % len(schedule)])
                actions, _ = policy(env.obs_buf, env.obs_history)
                env.step(actions)
                # Captured before reset, so falls and unloading remain visible.
                row = env.last_diagnostics[0]
                reset = int(env.reset_buf[0].item())
                failure = int(env.physical_failure[0].item())
                writer.writerow([step*env.dt, reset, failure]+row.cpu().tolist())
                accumulated += row.abs()
                count += 1
                resets += reset
                if step % int(1/env.dt) == 0:
                    print("t={:.1f}s target={:+.2f} cmd={:+.2f} vx={:+.3f} h={:.3f} resets={}".format(
                        step*env.dt, row[0].item(), row[1].item(), row[3].item(), row[8].item(), resets
                    ))
    with open(stem+".json", "w") as stream:
        json.dump({"frames": count, "env0_resets": resets, "stage": args.q1_stage,
                   "schedule_mps": schedule, "segment_s": 4.0,
                   "load_run": args.load_run, "checkpoint": args.checkpoint,
                   "mean_absolute_signals": dict(zip(DIAGNOSTIC_FIELDS, (accumulated/max(count, 1)).cpu().tolist()))}, stream, indent=2)
    print("Q1 evaluation CSV:", stem+".csv")
