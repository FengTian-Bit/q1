"""Q1 manual viewer playback or reproducible automatic evaluation."""
import csv
from datetime import datetime
import json
import os
import time

import torch
from isaacgym import gymapi

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils import export_policy_as_jit, task_registry
from .q1 import DIAGNOSTIC_FIELDS
from .keyboard import Q1Keyboard
from .q1_config import configure_q1_stage


def play_q1(args):
    cfg, train_cfg = task_registry.get_cfgs("q1")
    args.q1_stage = args.q1_stage or "turning"
    configure_q1_stage(cfg, args.q1_stage)
    cfg.env.num_envs = 16 if args.num_envs is None else args.num_envs
    cfg.init_state.reset_lin_vel = 0.0
    cfg.init_state.reset_ang_vel = 0.0
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.stand_probability = 1.0
    mode = getattr(args, "play_mode", "keyboard")
    if getattr(args, "headless", False):
        mode = "auto"
        print("Headless Q1 playback uses automatic commands.")
    duration = getattr(args, "play_duration", 0.0)
    if duration < 0:
        raise ValueError("--play_duration must be nonnegative")
    env, _ = task_registry.make_env("q1", args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    output = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported")
    export_policy_as_jit(runner.alg.actor_critic, os.path.join(output, "policies"))
    output = os.path.join(output, "evaluations")
    os.makedirs(output, exist_ok=True)
    stem = os.path.join(output, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    schedule = [(v, 0.0) for v in [0.0, 0.2, 0.0, -0.2, 0.0, 0.3, 0.0, -0.3, 0.0]]
    if args.q1_stage == "balance":
        schedule = [(0.0, 0.0)]
    elif args.q1_stage == "turning":
        schedule = [(0, 0), (0.2, 0), (0, 0), (0, 0.3), (0, 0.6), (0, 0),
                    (0, -0.3), (0, -0.6), (0, 0), (0.2, 0.3), (0.2, -0.3), (0, 0),
                    (-0.2, 0.3), (-0.2, -0.3), (0, 0), (0.3, 0.6), (-0.3, -0.6), (0, 0)]
    # Longer auto segments let a slow robot leave the central spawn platform.
    segment_s = 4.0 if cfg.terrain.mesh_type == "plane" else cfg.commands.rough_resampling_time
    if duration == 0 and mode == "auto":
        duration = max(72.0, segment_s*len(schedule))
    controls = Q1Keyboard(env) if mode == "keyboard" else None
    accumulated = torch.zeros(len(DIAGNOSTIC_FIELDS), device=env.device)
    count, resets = 0, 0
    try:
        if controls is not None:
            controls.enable(gymapi)
        with open(stem+".csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "env0_reset", "env0_failure"]+DIAGNOSTIC_FIELDS)
            while duration == 0 or count*env.dt < duration-1e-9:
                with torch.inference_mode():
                    if controls is not None:
                        controls.render()
                        if controls.paused:
                            time.sleep(0.02)
                            continue
                        controls.apply()
                    else:
                        env.set_command_target(*schedule[int(count*env.dt/segment_s) % len(schedule)])
                    actions, _ = policy(env.obs_buf, env.obs_history)
                    env.step(actions)
                    row = env.last_diagnostics[0]
                    reset = int(env.reset_buf[0].item())
                    failure = int(env.physical_failure[0].item())
                    writer.writerow([count*env.dt, reset, failure]+row.cpu().tolist())
                    accumulated += row.abs()
                    count += 1
                    resets += reset
                    if (count-1) % int(1/env.dt) == 0:
                        selected = controls.selected if controls is not None else 0
                        viewed = env.last_diagnostics[selected]
                        print("t={:.1f}s robot={} target={:+.2f} cmd={:+.2f} vx={:+.3f} yaw_target={:+.2f} yaw={:+.3f} h={:.3f} env0_resets={}".format(
                            (count-1)*env.dt, selected, viewed[0].item(), viewed[1].item(), viewed[3].item(),
                            env.command_targets[selected, 1].item(), viewed[5].item(), viewed[8].item(), resets
                        ))
    except SystemExit:
        # Viewer Esc/close still flushes diagnostic data and restores handlers.
        pass
    finally:
        if controls is not None:
            controls.close()
        with open(stem+".json", "w") as stream:
            json.dump({"frames": count, "env0_resets": resets, "stage": args.q1_stage,
                       "play_mode": mode, "schedule_mps": [pair[0] for pair in schedule] if mode == "auto" else None,
                       "schedule_vx_yaw": schedule if mode == "auto" else None,
                       "segment_s": segment_s if mode == "auto" else None,
                       "terrain_mesh": cfg.terrain.mesh_type,
                       "load_run": args.load_run, "checkpoint": args.checkpoint,
                       "mean_absolute_signals": dict(zip(DIAGNOSTIC_FIELDS, (accumulated/max(count, 1)).cpu().tolist()))}, stream, indent=2)
        print("Q1 evaluation CSV:", stem+".csv")
