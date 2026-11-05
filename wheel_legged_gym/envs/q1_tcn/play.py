"""Play a saved Q1 TCN policy with its own observation/control configuration."""

import argparse
import sys
import time

import isaacgym  # Must precede torch.
from isaacgym import gymapi
import numpy as np
import torch

import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args, task_registry
from .actor_critic import ActorCriticTCN
from .evaluate import scenario_config, set_evaluation_commands, command_at
from .runner import config_dict, load_checkpoint


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--mode', choices=['stand', 'tracking', 'turning', 'combined', 'keyboard'], default='keyboard')
    parser.add_argument('--duration', type=float, default=600.0)
    parser.add_argument('--tracking_speed', type=float, default=.25)
    parser.add_argument('--yaw_speed',type=float,default=1.)
    options, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        args = get_args(default_task='q1_tcn')
    finally:
        sys.argv = original
    for name, value in vars(options).items():
        setattr(args, name, value)
    if args.duration <= 0:
        raise ValueError('duration must be positive')
    return args


def main():
    return run(arguments())


def run(args, saved=None, scene_builder=None):
    torch.set_num_threads(1)
    saved = load_checkpoint(args.checkpoint_path) if saved is None else saved
    seed = args.seed if args.seed is not None else 201
    cfg, train_cfg = (scene_builder or scenario_config)(saved['config'], 'stand', seed,
                                     args.num_envs or 1, args.duration)
    if args.mode in ['tracking','combined'] and not 0 < args.tracking_speed <= min(-cfg.commands.ranges.lin_vel_x[0],cfg.commands.ranges.lin_vel_x[1]):
        raise ValueError('tracking_speed must be positive and within the trained forward/reverse range')
    if args.mode in ['turning','combined'] and not 0 < args.yaw_speed <= min(-cfg.commands.ranges.ang_vel_yaw[0],cfg.commands.ranges.ang_vel_yaw[1]):
        raise ValueError('yaw_speed must be positive and within the trained turning range')
    env, _ = task_registry.make_env('q1_tcn', args=args, env_cfg=cfg)
    try:
        model = ActorCriticTCN(env.num_obs, env.num_privileged_obs, env.num_actions,
                              ablation=train_cfg.ablation, **config_dict(train_cfg.policy)).to(args.rl_device)
        model.load_state_dict(saved['model_state_dict'])
        model.eval()
        env.evaluation_mode = env.external_commands = True
        target = [0.0, 0.0]
        if env.viewer and args.mode == 'keyboard':
            for key, action in [(gymapi.KEY_W, 'faster'), (gymapi.KEY_S, 'slower'),
                                (gymapi.KEY_A,'turn_left'),(gymapi.KEY_D,'turn_right'),
                                (gymapi.KEY_SPACE, 'stop')]:
                env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

            def handle_key(event):
                if event.value <= 0:
                    return
                if event.action == 'stop':
                    target[:] = [0.0, 0.0]
                elif event.action in ['turn_left','turn_right']:
                    lower,upper=cfg.commands.ranges.ang_vel_yaw
                    target[1]=float(np.clip(target[1]+(.1 if event.action=='turn_left' else -.1),lower,upper))
                elif event.action in ['faster', 'slower']:
                    target[0] = float(np.clip(target[0] + (.05 if event.action == 'faster' else -.05),
                                              *cfg.commands.ranges.lin_vel_x))
            env._extra_viewer_action_handler = handle_key
        set_evaluation_commands(env, 0.0)
        env.reset()
        if env.viewer:
            position = env.root_states[0, :3].cpu().numpy()
            env.set_camera(position + np.array([1.2, -1.2, .7]), position + np.array([0., 0., -.1]))
        print('Loaded {} | {} | obs={} | explicit velocity estimation={}'.format(
            args.checkpoint_path, train_cfg.ablation, env.num_obs, model.enabled[0]), flush=True)
        print('W/S: vx +/-0.05 within {}; A/D: yaw +/-0.1 within {}; Space: stop; Esc: exit'.format(
            cfg.commands.ranges.lin_vel_x,cfg.commands.ranges.ang_vel_yaw), flush=True)
        print('Linear command acceleration limit: {} m/s^2 (0 means legacy step commands)'.format(
            cfg.commands.max_linear_acceleration),flush=True)
        start = time.time()
        for step in range(round(args.duration / env.dt)):
            if args.mode in ['tracking','turning','combined']:
                direction=command_at((step*env.dt)%12,'tracking')/.25
                target[:]=[direction*args.tracking_speed if args.mode!='turning' else 0.,
                           direction*args.yaw_speed if args.mode!='tracking' else 0.]
            elif args.mode == 'stand':
                target[:] = [0.0, 0.0]
            set_evaluation_commands(env, target[0], target[1])
            env.obs_buf[:, 6:9] = env.commands[:, :3] * env.commands_scale
            env.obs_history[:, -env.num_obs+6:-env.num_obs+9] = env.obs_buf[:, 6:9]
            with torch.no_grad():
                action, _ = model.act_inference(env.obs_buf.to(args.rl_device), env.obs_history.to(args.rl_device))
                env.step(action.to(env.device))
            if step % max(1, round(1 / env.dt)) == 0:
                metric = env.transition_metrics
                print('t={:.1f}s target={:+.2f}/{:+.2f} cmd={:+.2f}/{:+.2f} vx/wz={:+.3f}/{:+.3f} pitch/roll={:+.2f}/{:+.2f}deg park={:.2f}cm'.format(
                    (step+1)*env.dt, target[0], target[1],float(metric['command'][0]),float(metric['yaw_command'][0]),float(metric['velocity'][0, 0]),
                    float(metric['yaw_rate'][0]),
                    float(metric['pitch'][0])*180/np.pi, float(metric['roll'][0])*180/np.pi,
                    float(metric['parking_error'][0])*100), flush=True)
            if env.transition_metrics['failure'][0]:
                print('Robot 0 failed; stopping playback.', flush=True)
                break
        print('Playback finished in {:.1f}s'.format(time.time() - start), flush=True)
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
