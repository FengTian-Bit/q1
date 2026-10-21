"""Deterministic Q1 evaluation, without wl height/speed or PI compensation."""

import os

import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils import export_policy_as_jit, task_registry


def play_q1(args):
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = 16 if args.num_envs is None else args.num_envs
    cfg.noise.add_noise = False
    cfg.init_state.reset_lin_vel = 0.0
    cfg.init_state.reset_ang_vel = 0.0
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    cfg.commands.stand_probability = 1.0
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    path = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "logs", "q1_flat", "exported", "policies")
    export_policy_as_jit(runner.alg.actor_critic, path)
    schedule = [0.0] if args.q1_stage == "balance" else [0.0, 0.2, 0.0, -0.2, 0.0]
    print("Q1 evaluation speeds (m/s, 4 s each):", schedule)
    for step in range(int(60.0 / env.dt)):
        with torch.inference_mode():
            speed = schedule[int(step * env.dt / 4.0) % len(schedule)]
            env.commands[:, 0] = speed
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = cfg.commands.ranges.height[0]
            # Update command observation before inference; do not append a second frame.
            env.obs_buf[:] = env.compute_proprioception_observations()
            env.obs_history[:, -env.num_obs:] = env.obs_buf
            if runner.alg.actor_critic.is_sequence:
                actions, _ = policy(env.obs_buf, env.obs_history)
            else:
                actions = policy(env.obs_buf)
            _, _, _, dones, _, _ = env.step(actions)
        if step % int(1.0 / env.dt) == 0:
            print("t={:.1f}s cmd={:+.2f} vx={:+.3f} height={:.3f} resets={}".format(
                step * env.dt, speed, env.base_lin_vel[0, 0].item(),
                env.base_height[0].item(), dones.sum().item()
            ))
