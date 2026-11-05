"""One uninterrupted training run at the full requested command range."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import isaacgym  # before torch
import torch
import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args, task_registry
from .runner import TCNPpoRunner, config_dict
from tools.q1_reward_iteration import training_config, warm_start_weights, inherit_optimizer


def initialize_yaw_inputs(runner):
    """Unused yaw-command columns start at zero, preserving all zero-yaw actions."""
    model=runner.model
    columns=[(model.input_projection.weight,7),(model.actor[0].weight,7),
             (model.critic[0].weight,10)]
    if model.wheel_residual is not None:columns.append((model.wheel_residual[0].weight,7))
    with torch.no_grad():
        for parameter,column in columns:
            parameter[:,column]=0
            state=runner.alg.optimizer.state.get(parameter,{})
            for key in ['exp_avg','exp_avg_sq']:
                if key in state:state[key][:,column]=0


def evaluate(checkpoint, output, args, seeds, screen):
    command=[sys.executable,'-u','-m','wheel_legged_gym.envs.q1_tcn.target_evaluate',
        '--checkpoint_path',str(checkpoint),'--output_dir',str(output),'--stage',args.stage,
        '--seeds']+[str(seed) for seed in seeds]+[
        '--num_envs',str(32 if screen else 64),'--headless',
        '--sim_device',args.sim_device,'--rl_device',args.rl_device]
    if screen:command.append('--screen')
    with output.with_suffix('.log').open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return json.loads((output/'summary.json').read_text())


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage',choices=['linear','turning'],required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--warm_start',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--screen_interval',type=int,default=250)
    parser.add_argument('--continue_training',action='store_true',
                        help='Restore counters/optimizer/RNG; other changes need explicit update flags')
    parser.add_argument('--update_tracking_reward',action='store_true',
                        help='Explicitly allow only reward v7 tracking normalization on continuation')
    parser.add_argument('--update_yaw_reward_weight',action='store_true',
                        help='Only set angular tracking weight to half the unchanged linear weight')
    parser.add_argument('--update_wheel_diff_reward',action='store_true',
                        help='Explicitly enable only the wheel differential guidance reward on continuation')
    opts,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining;args=get_args(default_task='q1_tcn')
    for key,value in vars(opts).items():setattr(args,key,value)
    args.inherit_checkpoint_config=args.inherit_optimizer=True
    if args.update_tracking_reward and not args.continue_training:
        raise ValueError('--update_tracking_reward requires --continue_training')
    if args.update_yaw_reward_weight and not args.continue_training:
        raise ValueError('--update_yaw_reward_weight requires --continue_training')
    if args.update_wheel_diff_reward and not args.continue_training:
        raise ValueError('--update_wheel_diff_reward requires --continue_training')
    torch.set_num_threads(1)
    cfg,ppo,spec=training_config(args)
    if args.screen_interval<1 or ppo.runner.max_iterations<1:
        raise ValueError('Training and screen intervals must be positive')
    if spec.get('performance_curriculum') or spec.get('speed_curriculum'):
        raise ValueError('Target runs never change command ranges by iteration')
    if not cfg.commands.target_mode or cfg.commands.target_stage!=args.stage:
        raise ValueError('Target command sampling must match the requested stage')
    if cfg.commands.ranges.lin_vel_x!=[-1.2,1.2]:raise ValueError('Use the complete +/-1.2 range from iteration one')
    if args.stage=='turning' and cfg.commands.ranges.ang_vel_yaw!=[-1.,1.]:raise ValueError('Turning stage must cover +/-1rad/s from iteration one')
    if args.output_dir.exists():raise ValueError('Use a new run folder; previous models are immutable')
    env,_=task_registry.make_env('q1_tcn',args=args,env_cfg=cfg)
    start=time.time();selected=None;holdout_attempt=0
    try:
        runner=TCNPpoRunner(env,ppo,args.output_dir,device=args.rl_device)
        if args.continue_training:
            runner.load(args.warm_start,allow_symmetry_update=True,
                        allow_tracking_update=args.update_tracking_reward,
                        allow_yaw_weight_update=args.update_yaw_reward_weight,
                        allow_wheel_diff_update=args.update_wheel_diff_reward)
        else:
            warm_start_weights(runner.model,args.warm_start,cfg.odometry.observe_wheel_increments)
            inherit_optimizer(runner,args.warm_start)
        if 'motion_noise_std' in spec and not args.continue_training:
            std=float(spec['motion_noise_std'])
            if not 0<std<=ppo.policy.max_noise_std:raise ValueError('Invalid motion exploration std')
            with torch.no_grad():runner.model.log_std.fill_(math.log(std))
        if spec.get('initialize_yaw_inputs') and not args.continue_training:initialize_yaw_inputs(runner)
        # Match train.py: populate reset joint states, commands and history
        # before PPO collects its first transition.
        env.reset()
        manifest=dict(stage=args.stage,warm_start=str(args.warm_start.resolve()),
            warm_start_sha256=hashlib.sha256(args.warm_start.read_bytes()).hexdigest(),spec=spec,
            command_ranges_from_iteration_one=config_dict(cfg.commands.ranges),
            optimizer_inherited=True,training_seed=ppo.seed,num_envs=env.num_envs,
            continued_from_iteration=runner.iteration,continued_from_total_steps=runner.total_steps,
            symmetry_loss_coef=ppo.algorithm.symmetry_loss_coef,
            tracking_reward_updated=args.update_tracking_reward,
            yaw_reward_weight_updated=args.update_yaw_reward_weight,
            wheel_diff_reward_updated=args.update_wheel_diff_reward,
            environment_reset_before_first_rollout=True,
            explicit_velocity_estimation=False,zero_yaw_inputs_initialized=bool(spec.get('initialize_yaw_inputs')) and not args.continue_training,
            parking_policy_frozen=all(not p.requires_grad for p in runner.model.parking_policy.parameters()))
        (args.output_dir/'initialization.json').write_text(json.dumps(manifest,indent=2))
        (args.output_dir/'screens').mkdir();(args.output_dir/'holdouts').mkdir()
        checkpoint=runner.save()
        while runner.iteration<ppo.runner.max_iterations:
            runner.learn(min(args.screen_interval,ppo.runner.max_iterations-runner.iteration))
            checkpoint=runner.save();label='iter{:04d}'.format(runner.iteration)
            screen=evaluate(checkpoint,args.output_dir/'screens'/label,args,[1111],True)
            record=dict(iteration=runner.iteration,stage=args.stage,screen_passed=screen['passed'],
                failed_cases=[dict(case=row['case'],seed=row['seed'],checks=[key for key,value in row['checks'].items() if not value]) for row in screen['cases'] if not row['passed']])
            print('TARGET SCREEN',json.dumps(record),flush=True)
            with (args.output_dir/'progress.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if not screen['passed']:continue
            seed_start=(1201 if args.stage=='linear' else 1301)+10*holdout_attempt
            holdout_attempt+=1
            output=args.output_dir/'holdouts'/label
            holdout=evaluate(checkpoint,output,args,list(range(seed_start,seed_start+3)),False)
            print('TARGET HOLDOUT',runner.iteration,'passed',holdout['passed'],flush=True)
            if holdout['passed']:
                selected=dict(checkpoint=str(checkpoint.resolve()),evaluation=str(output.resolve()))
                break
        result=dict(status='goal_passed' if selected else 'budget_exhausted',stage=args.stage,
            iteration=runner.iteration,checkpoint=selected['checkpoint'] if selected else str(checkpoint.resolve()),
            evaluation=selected['evaluation'] if selected else None,accuracy_goal=.90,
            elapsed_seconds=time.time()-start,one_training_process=True,
            fixed_command_ranges=config_dict(cfg.commands.ranges))
        path=args.output_dir/'target_result.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2));temp.replace(path)
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
