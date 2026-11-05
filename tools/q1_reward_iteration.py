"""One-GPU reward pilots and per-joint diagnostics; all overrides are saved."""

import argparse
import json
import csv
import sys
import time
import subprocess
from collections import deque
from pathlib import Path

import isaacgym  # before torch
from isaacgym import gymapi
import numpy as np
import torch

import wheel_legged_gym.envs  # register tasks before importing the registry
from wheel_legged_gym.utils import get_args, task_registry
from wheel_legged_gym.envs.q1_tcn.q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO, validate_config
from wheel_legged_gym.envs.q1_tcn.runner import TCNPpoRunner, config_dict, restore_config, load_checkpoint
from wheel_legged_gym.envs.q1_tcn.actor_critic import ActorCriticTCN
from wheel_legged_gym.envs.q1_tcn.evaluate import scenario_config, set_evaluation_commands, command_at
from wheel_legged_gym.envs.q1_tcn.metrics import TrialMetrics, summarize
from wheel_legged_gym.envs.q1_tcn.curriculum import tracking_gate
from wheel_legged_gym.envs.q1_tcn.command_reference import command_reference_inputs


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=['train', 'diagnose'], required=True)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--checkpoint_path', type=Path)
    parser.add_argument('--warm_start', type=Path)
    parser.add_argument('--inherit_checkpoint_config', action='store_true',
                        help='Restore frozen environment/policy/PPO settings before applying spec and CLI')
    parser.add_argument('--inherit_optimizer', action='store_true')
    parser.add_argument('--tracking_speed', type=float, default=.25)
    parser.add_argument('--command_reference_limit',type=float,default=None,
                        help='Fixed-weight diagnostic of command-derived wheel feedforward; no training')
    parser.add_argument('--reference_action_limit',type=float,default=None,
                        help='Untrained diagnostic only: normalized wheel action bound for reference probe')
    parser.add_argument('--reference_ramp_rate',type=float,default=None,
                        help='Diagnostic internal command slew rate m/s^2; reward/metrics keep actual requested steps')
    parser.add_argument('--scenario', default='stand', choices=['stand','tracking','push_front','push_side','payload'])
    parser.add_argument('--duration', type=float, default=12)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--capture_collision', action='store_true')
    parser.add_argument('--probe_odometry', action='store_true')
    parser.add_argument('--trace_failures',action='store_true',help='Save the last two seconds of inputs and controller outputs before each first fall')
    parser.add_argument('--policy_probe_spec',type=Path,
                        help='Explicit untrained policy-setting diagnostic; model parameter layout must still match')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    args = get_args(default_task='q1_tcn')
    for key, value in vars(options).items():
        setattr(args, key, value)
    return args


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def warm_start_weights(model, checkpoint, observe_wheel_increments=False):
    saved=load_checkpoint(checkpoint)
    state=dict(saved['model_state_dict'])
    old_obs=state['input_projection.weight'].shape[1]
    if model.num_obs != old_obs:
        if model.num_obs-old_obs != 3:
            raise ValueError('Only the explicit 3-slot odometry extension can be warm-started')
        current=model.state_dict()
        for name,insert in [('input_projection.weight',old_obs),('actor.0.weight',old_obs),
                            ('critic.0.weight',old_obs+3)]:
            old=state[name]
            value=torch.zeros_like(current[name],device='cpu')
            value[:,:insert]=old[:,:insert]
            if name!='input_projection.weight':value[:,insert+3:]=old[:,insert:]
            state[name]=value
    previous=saved.get('config',{}).get('env',{}).get('odometry',{})
    if observe_wheel_increments and not previous.get('observe_wheel_increments',False):
        # These inputs were always zero in the old interface. Zero their
        # random unused weights so activating sensors preserves the policy.
        for name in ['input_projection.weight','actor.0.weight','critic.0.weight']:
            state[name]=state[name].clone()
        state['input_projection.weight'][:,[11,14],:]=0
        state['actor.0.weight'][:,[11,14]]=0
        state['critic.0.weight'][:,[14,17]]=0
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith(('wheel_residual.','parking_policy.')) for name in missing):
        raise ValueError('Unexpected warm-start parameter mismatch: {} / {}'.format(missing,unexpected))
    model.constrain_distribution_parameters()
    return old_obs


def record_wheel_residual(row, residual, active, limit, command_scale):
    """Record already converted NumPy estimates; never touch policy tensors."""
    values=residual[active]
    bound=limit*min(1.,abs(row['command_m_s']*command_scale))
    for index,side in enumerate(['L','R']):
        row['wheel_residual_'+side]=float(values[:,index].mean())
        row['wheel_residual_sat_'+side]=float((np.abs(values[:,index])>=.99*bound).mean()) if bound>0 else 0.


def training_config(args):
    cfg, ppo = Q1TCNCfg(), Q1TCNCfgPPO()
    ppo.ablation='A0'
    if args.inherit_checkpoint_config:
        if not args.warm_start:
            raise ValueError('--inherit_checkpoint_config requires --warm_start')
        saved=load_checkpoint(args.warm_start)['config']
        restore_config(cfg,saved['env'])
        restore_config(ppo,saved['train'])
    else:
        cfg.env.num_envs=2048
        ppo.runner.max_iterations=1000
    spec=json.loads(args.spec.read_text()) if args.spec else {}
    restore_config(cfg, spec.get('env', {}))
    for section in ['policy','algorithm','runner']:
        restore_config(getattr(ppo,section), spec.get(section,{}))
    for section,flag in [('rewards','freeze_rewards'),('control','freeze_control')]:
        if spec.get(flag):
            if not args.inherit_checkpoint_config or config_dict(getattr(cfg,section)) != saved['env'][section]:
                raise ValueError('Frozen {} differ from accepted checkpoint'.format(section))
    cfg.env.num_envs=args.num_envs if args.num_envs is not None else cfg.env.num_envs
    cfg.seed=ppo.seed=args.seed if args.seed is not None else ppo.seed
    if args.max_iterations is not None:
        ppo.runner.max_iterations=args.max_iterations
    ppo.runner.save_interval=100
    if ppo.ablation != 'A0':
        raise ValueError('Reward pilots must use A0 with explicit velocity estimation disabled')
    validate_config(cfg,ppo)
    return cfg,ppo,spec


def speed_curriculum(spec, target_range, iterations):
    stages=spec.get('speed_curriculum',[])
    if not stages:
        return []
    starts=[stage[0] for stage in stages]
    limits=[stage[1] for stage in stages]
    target=min(-target_range[0],target_range[1])
    if target_range[0] != -target_range[1]:
        raise ValueError('Speed curriculum requires a symmetric target range')
    if starts[0] != 0 or any(not isinstance(start,int) for start in starts):
        raise ValueError('Curriculum starts must be integer iterations, beginning at zero')
    if any(a>=b for a,b in zip(starts,starts[1:])) or starts[-1]>=iterations:
        raise ValueError('Curriculum starts must increase and fit inside the training budget')
    if any(not 0<limit<=target for limit in limits) or any(a>=b for a,b in zip(limits,limits[1:])):
        raise ValueError('Curriculum speeds must increase and remain within the target range')
    if limits[-1] != target:
        raise ValueError('Final curriculum speed must equal the target range')
    return [(start,limit) for start,limit in stages]


def apply_speed_limit(env, snapshot, limit, edge_fraction=0):
    ranges=env.command_ranges['lin_vel_x']
    if ranges.shape != (env.num_envs,2):
        raise ValueError('Expected one speed range tensor row per environment')
    ranges[:,0]=-limit
    ranges[:,1]=limit
    if not 0<=edge_fraction<=1:
        raise ValueError('edge_fraction must be in [0, 1]')
    count=int(env.num_envs*edge_fraction/2)
    if count:
        ids=torch.randperm(env.num_envs,device=env.device)
        ranges[ids[:count],0]=.85*limit
        ranges[ids[count:2*count],1]=-.85*limit
    current_range=[-limit,limit]
    env.cfg.commands.ranges.lin_vel_x=current_range
    snapshot['env']['commands']['ranges']['lin_vel_x']=current_range
    return current_range


def inherit_optimizer(runner, checkpoint):
    saved=load_checkpoint(checkpoint)
    if saved['model_state_dict']['input_projection.weight'].shape[1] != runner.model.num_obs:
        raise ValueError('Optimizer transfer requires identical observation dimensions')
    previous=saved['optimizer_state_dict']
    current=runner.alg.optimizer.state_dict()
    old_ids=previous['param_groups'][0]['params']
    new_ids=current['param_groups'][0]['params']
    if len(old_ids)!=len(new_ids):
        old_names=list(saved['model_state_dict'])
        new_names=list(dict(runner.model.named_parameters()))
        if len(previous['param_groups'])!=1 or len(old_names)!=len(old_ids):
            raise ValueError('Cannot map legacy optimizer parameters unambiguously')
        if new_names[:len(old_names)]!=old_names or any(not name.startswith(('wheel_residual.','parking_policy.')) for name in new_names[len(old_names):]):
            raise ValueError('Only appended residual/parking parameters may extend the optimizer')
        # All inherited parameter IDs stay aligned; new head moments start empty.
        previous['param_groups'][0]['params']=new_ids
    runner.alg.optimizer.load_state_dict(previous)
    # Keep the newly chosen fine-tuning rate; inherit Adam moments and steps.
    runner.alg.learning_rate=runner.cfg.algorithm.learning_rate
    for group in runner.alg.optimizer.param_groups:
        group['lr']=runner.alg.learning_rate


def evaluate_gate(args, checkpoint, output, scenario, limit, seed, num_envs=64, duration=12):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--mode','diagnose',
             '--checkpoint_path',str(checkpoint),'--output_dir',str(output),
             '--scenario',scenario,'--tracking_speed',str(limit),'--duration',str(duration),
             '--num_envs',str(num_envs),'--seed',str(seed),'--headless',
             '--sim_device',args.sim_device,'--rl_device',args.rl_device]
    with output.with_suffix('.log').open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return json.loads((output/'diagnostic.json').read_text())


def learn_with_gate(runner, args, settings):
    gate_root=args.output_dir/'gates'
    gate_root.mkdir()
    limit=settings['initial_limit']
    target=settings['target_limit']
    consecutive=0
    while runner.iteration<runner.cfg.runner.max_iterations:
        apply_speed_limit(runner.env,runner.snapshot,limit,settings.get('edge_fraction',0))
        (args.output_dir/'resolved_config.json').write_text(json.dumps(runner.snapshot,indent=2))
        count=min(settings['chunk_iterations'],runner.cfg.runner.max_iterations-runner.iteration)
        print('Gated training: iter={} limit={:.3f}'.format(runner.iteration,limit),flush=True)
        runner.learn(count)
        checkpoint=runner.save()
        prefix='iter{:04d}'.format(runner.iteration)
        evaluation=dict(num_envs=settings.get('eval_num_envs',64),duration=settings.get('eval_duration',12))
        tracking=evaluate_gate(args,checkpoint,gate_root/(prefix+'_tracking'),'tracking',limit,settings['seed'],**evaluation)
        standing=evaluate_gate(args,checkpoint,gate_root/(prefix+'_stand'),'stand',limit,settings['seed'],**evaluation)
        passed,checks=tracking_gate(tracking,standing,settings['thresholds'])
        # End-point success must also preserve the requested lower-speed cases.
        for speed in settings.get('additional_tracking_speeds', []):
            if speed >= limit:
                continue
            label='speed{:03d}'.format(round(speed*1000))
            lower=evaluate_gate(args,checkpoint,gate_root/(prefix+'_'+label),
                                'tracking',speed,settings['seed'],**evaluation)
            lower_passed,lower_checks=tracking_gate(lower,standing,settings['thresholds'])
            checks.update({label+'_'+key:value for key,value in lower_checks.items()})
            passed=passed and lower_passed
        record=dict(iteration=runner.iteration,limit=limit,passed=passed,checks=checks,checkpoint=str(checkpoint))
        with (args.output_dir/'gate_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('Gate {} limit={:.3f} passed={} failures={}'.format(
            runner.iteration,limit,passed,[key for key,value in checks.items() if not value]),flush=True)
        if passed and limit>=target-1e-6:
            consecutive+=1
            if consecutive>=settings.get('target_passes',2):
                print('Target range passed repeated screens; independent holdout still required',flush=True)
                break
        elif passed:
            limit=min(target,round(limit+settings['increment'],6))
        else:
            consecutive=0
    result=dict(status='screen_passed_holdout_required' if consecutive>=settings.get('target_passes',2) else 'budget_exhausted',
                iteration=runner.iteration,limit=runner.env.cfg.commands.ranges.lin_vel_x[1],
                target=target,checkpoint=str(runner.save()))
    (args.output_dir/'performance_result.json').write_text(json.dumps(result,indent=2))


def train(args):
    cfg,ppo,spec=training_config(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new output directory for a new training experiment')
    target_range=list(cfg.commands.ranges.lin_vel_x)
    stages=speed_curriculum(spec,target_range,ppo.runner.max_iterations)
    gated=spec.get('performance_curriculum')
    if gated:
        if stages or not 0<gated['initial_limit']<=gated['target_limit']<=min(-target_range[0],target_range[1]):
            raise ValueError('Gated curriculum must fit target range and cannot combine with timed stages')
        if gated['chunk_iterations']<1 or gated['increment']<=0 or gated.get('target_passes',2)<1:
            raise ValueError('Gated curriculum needs positive chunk, increment and target passes')
        if any(not 0 < speed <= gated['target_limit'] for speed in gated.get('additional_tracking_speeds', [])):
            raise ValueError('Additional tracking speeds must fit the target range')
        cfg.commands.ranges.lin_vel_x=[-gated['initial_limit'],gated['initial_limit']]
    if stages:
        cfg.commands.ranges.lin_vel_x=[-stages[0][1],stages[0][1]]
    env,_=task_registry.make_env('q1_tcn', args=args, env_cfg=cfg)
    start=time.time()
    try:
        runner=TCNPpoRunner(env,ppo,args.output_dir,device=args.rl_device)
        if args.warm_start:
            old_obs=warm_start_weights(runner.model,args.warm_start,cfg.odometry.observe_wheel_increments)
        else:old_obs=env.num_obs
        parking_checkpoint=spec.get('parking_warm_start')
        if runner.model.parking_policy is not None:
            inherited=args.warm_start and any(name.startswith('parking_policy.') for name in load_checkpoint(args.warm_start)['model_state_dict'])
            if parking_checkpoint:
                warm_start_weights(runner.model.parking_policy,parking_checkpoint)
            elif not inherited:
                raise ValueError('A new frozen parking branch requires parking_warm_start')
        elif parking_checkpoint:
            raise ValueError('parking_warm_start requires policy.parking_policy')
        if args.inherit_optimizer:
            if not args.warm_start:raise ValueError('--inherit_optimizer requires --warm_start')
            inherit_optimizer(runner,args.warm_start)
        manifest=dict(spec=spec,warm_start=str(args.warm_start) if args.warm_start else None,
                      inherit_checkpoint_config=args.inherit_checkpoint_config,
                      target_speed_range=target_range,speed_curriculum=stages,
                      fresh_optimizer=not args.inherit_optimizer,training_seed=ppo.seed,initial_num_obs=old_obs,
                      wheel_increment_observations=cfg.odometry.observe_wheel_increments,
                      num_obs=env.num_obs,velocity_estimation_enabled=False)
        manifest['parking_warm_start']=parking_checkpoint
        (args.output_dir/'iteration_manifest.json').write_text(json.dumps(manifest,indent=2))
        env.reset()
        if gated:
            learn_with_gate(runner,args,gated)
        elif stages:
            for index,(start_iteration,limit) in enumerate(stages):
                end_iteration=stages[index+1][0] if index+1<len(stages) else ppo.runner.max_iterations
                current_range=apply_speed_limit(env,runner.snapshot,limit)
                (args.output_dir/'resolved_config.json').write_text(json.dumps(runner.snapshot,indent=2))
                print('Speed curriculum: iter {}..{} range {}'.format(start_iteration,end_iteration,current_range),flush=True)
                runner.learn(end_iteration-start_iteration)
        else:
            runner.learn(ppo.runner.max_iterations)
        print('elapsed_seconds',time.time()-start,flush=True)
    finally:
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


def diagnose(args):
    saved=load_checkpoint(args.checkpoint_path)
    seed=args.seed if args.seed is not None else 101
    cfg,ppo=scenario_config(saved['config'], args.scenario,seed,args.num_envs or 64,args.duration)
    policy_probe=json.loads(args.policy_probe_spec.read_text()) if args.policy_probe_spec else {}
    if policy_probe:
        restore_config(ppo.policy,policy_probe['policy'])
    if args.command_reference_limit is not None and ppo.policy.command_reference:
        raise ValueError('Checkpoint already has command reference; diagnostic override would apply it twice')
    if args.reference_action_limit is not None:
        if args.command_reference_limit is None or not 1<=args.reference_action_limit<=2:
            raise ValueError('Reference action limit requires an explicit reference probe and must be in [1, 2]')
        cfg.normalization.clip_actions=args.reference_action_limit
    if args.reference_ramp_rate is not None and (args.command_reference_limit is None or args.reference_ramp_rate<=0):
        raise ValueError('Positive reference ramp rate requires an explicit reference probe')
    if args.spec:
        spec=json.loads(args.spec.read_text())
        restore_config(cfg,spec.get('env',{}))
    validate_config(cfg,ppo)
    if args.scenario == 'tracking':
        lower,upper=cfg.commands.ranges.lin_vel_x
        if args.command_reference_limit is not None:
            if not 0<args.command_reference_limit<=min(-lower,upper):
                raise ValueError('Command reference must remain within the saved training range')
            cfg.commands.ranges.lin_vel_x=[-max(-lower,args.tracking_speed),max(upper,args.tracking_speed)]
        elif not 0 < args.tracking_speed <= min(-lower,upper):
            raise ValueError('Tracking speed must be positive and within the trained forward/reverse range')
    if args.capture or args.capture_collision:
        cfg.env.enable_camera_sensors=True
    env,_=task_registry.make_env('q1_tcn',args=args,env_cfg=cfg)
    try:
        model=ActorCriticTCN(env.num_obs,env.num_privileged_obs,6,ablation=ppo.ablation,
                             **config_dict(ppo.policy)).to(args.rl_device)
        model.load_state_dict(saved['model_state_dict']);model.eval()
        env.evaluation_mode=env.external_commands=True
        set_evaluation_commands(env,0);env.reset()
        odometry=None
        if args.probe_odometry:
            from wheel_legged_gym.envs.q1_tcn.odometry import WheelOdometry
            odometry=WheelOdometry(env.num_envs,env.device,radius=cfg.asset.wheel_radius,
                                    l1=cfg.asset.l1,l2=cfg.asset.l2)
            def encoder_imu():
                q=env.dof_pos[:,env.policy_dof_indices].clone()
                q[:,[0,1,3,4]]=(env.obs_buf[:,9:15]/env.obs_scales.dof_pos+
                    env.default_dof_pos[:,env.policy_dof_indices])[:,[0,1,3,4]]
                if env.add_noise:
                    q[:,[2,5]]+=(2*torch.rand(env.num_envs,2,device=env.device)-1)*cfg.noise.noise_level*cfg.noise.noise_scales.dof_pos
                return q,env.obs_buf[:,:3]/env.obs_scales.ang_vel,env.obs_buf[:,3:6]
            q,gyro,gravity=encoder_imu()
            odometry.reset(torch.arange(env.num_envs,device=env.device),q,gravity)
            true_start=env.root_states[:,:2].clone()  # diagnostic target, never an observer input
        camera=None
        if args.capture or args.capture_collision:
            properties=gymapi.CameraProperties();properties.width=960;properties.height=720
            properties.use_collision_geometry=args.capture_collision
            camera=env.gym.create_camera_sensor(env.envs[0],properties)
            if camera < 0:raise RuntimeError('Unable to create diagnostic camera')
            origin=env.root_states[0,:3].cpu().numpy()
            env.gym.set_camera_location(camera,env.envs[0],
                gymapi.Vec3(float(origin[0]+1),float(origin[1]-1),float(origin[2]+.5)),
                gymapi.Vec3(float(origin[0]),float(origin[1]),float(origin[2]-.1)))
            args.output_dir.mkdir(parents=True,exist_ok=True)
        push_end=3.15 if args.scenario.startswith('push_') else None
        metrics=TrialMetrics(env.num_envs,env.dt,env.root_states[:,:3].cpu().numpy(),model.enabled,push_end)
        trace=[];joint_sums={k:np.zeros(6) for k in ['unbounded','raw','raw_sq','executed','clip','near_limit','position','target','velocity','torque']}
        samples=0; signed_pitch=0;signed_roll=0
        failure_windows=[];recent_steps=deque(maxlen=max(1,round(2/env.dt)))
        reference_history=torch.zeros(env.num_envs,ppo.policy.history_length,device=args.rl_device)
        phase_sums={phase:{key:np.zeros(env.num_envs) for key in ['count','sq','bias']}
                    for phase in ['forward','reverse','stop']}
        capture_times = [2., 4.5, 9.5] if args.scenario == 'tracking' else [2.]
        capture_steps = {round(time_s/env.dt) for time_s in capture_times}
        capture_steps.add(round(args.duration/env.dt)-1)
        for step in range(round(args.duration/env.dt)):
            t=step*env.dt
            set_evaluation_commands(env,command_at(t,args.scenario)*args.tracking_speed/.25)
            env.obs_buf[:,6:9]=env.commands[:,:3]*env.commands_scale
            env.obs_history[:,-env.num_obs+6:-env.num_obs+9]=env.obs_buf[:,6:9]
            env.eval_force.zero_()
            if push_end and 3<=t<push_end:
                env.eval_force[:,1 if args.scenario=='push_side' else 0]=8
            with torch.no_grad():
                if args.trace_failures:
                    before_obs=env.obs_buf.clone()
                    before_history=env.obs_history.clone()
                policy_obs,policy_history=env.obs_buf.to(args.rl_device),env.obs_history.to(args.rl_device)
                offset=None
                if args.command_reference_limit is not None:
                    if args.reference_ramp_rate is not None:
                        requested=env.commands[:,0].to(args.rl_device)
                        maximum=args.reference_ramp_rate*env.dt
                        current=reference_history[:,-1]+(requested-reference_history[:,-1]).clamp(-maximum,maximum)
                        reference_history=torch.cat((reference_history[:,1:],current[:,None]),dim=-1)
                        policy_obs=policy_obs.clone();policy_history=policy_history.clone()
                        policy_obs[:,6]=current*env.commands_scale[0]
                        frames=policy_history.reshape(env.num_envs,-1,env.num_obs)
                        frames[:,:,6]=reference_history*env.commands_scale[0]
                    policy_obs,policy_history,offset=command_reference_inputs(
                        policy_obs,policy_history,args.command_reference_limit,float(env.commands_scale[0]),
                        cfg.asset.wheel_radius,cfg.control.vel_action_scale,env.obs_scales.dof_vel,
                        cfg.odometry.increment_scale if cfg.odometry.observe_wheel_increments else None,env.dt)
                action,estimate=model.act_inference(policy_obs,policy_history)
                if offset is not None:
                    action=action.clone()
                    action[:,[2,5]]+=offset[:,None]
                    bound=args.reference_action_limit or model.mean_limit
                    if bound is not None:
                        action[:,[2,5]]=action[:,[2,5]].clamp(-bound,bound)
                estimates={k:v.cpu().numpy() for k,v in estimate.items()}
                estimates['contact_probability']=estimate['contact_logits'].sigmoid().cpu().numpy()
                metrics.update_estimates(estimates,env.estimator_targets.cpu().numpy())
                env.step(action.to(env.device))
            sample={k:v.cpu().numpy() for k,v in env.transition_metrics.items()}
            if odometry is not None:
                q,gyro,gravity=encoder_imu()
                odometry.update(q,gyro,gravity,env.dt)
                odometry_error=(odometry.xy-(env.root_states[:,:2]-true_start)).norm(dim=-1)
            active=metrics.active.copy()
            if args.trace_failures:
                recent_steps.append(dict(time_s=t+env.dt,obs=before_obs.cpu().numpy().copy(),
                    action=action.cpu().numpy().copy(),
                    motion_mean=estimates.get('motion_mean',action.cpu().numpy()).copy(),
                    parking_mean=estimates.get('parking_mean',np.zeros((env.num_envs,6))).copy(),
                    parking_weight=estimates.get('parking_weight',np.zeros(env.num_envs)).copy(),
                    terminal_critic=env.terminal_critic_obs.cpu().numpy().copy()))
                for index in np.flatnonzero(active & sample['failure']):
                    window=[]
                    for item in recent_steps:
                        window.append({key:value if key=='time_s' else value[index].tolist()
                                       for key,value in item.items()})
                    failure_windows.append(dict(trial=int(index),failure_time_s=t+env.dt,
                        history_before_failure=before_history[index].cpu().tolist(),steps=window))
            if args.scenario == 'tracking':
                phase='forward' if 4<=t<5 else 'reverse' if 9<=t<10 else 'stop' if 11<=t<12 else None
                if phase is not None:
                    sums=phase_sums[phase]
                    error=sample['velocity'][:,0]-sample['command']
                    sums['count']+=active
                    sums['sq']+=np.where(active,error**2,0.)
                    sums['bias']+=np.where(active,error,0.)
            raw=action.cpu().numpy()
            limits=np.full(6,cfg.normalization.clip_actions)
            if cfg.normalization.leg_clip_actions is not None:
                limits[[0,1,3,4]]=cfg.normalization.leg_clip_actions
            data=dict(unbounded=estimates['action_mean_unbounded'],raw=raw,raw_sq=raw**2,
                executed=env.actions.cpu().numpy(),clip=np.abs(raw)>limits,near_limit=np.abs(raw)>.9*limits,
                position=env.dof_pos[:,env.policy_dof_indices].cpu().numpy(),
                target=env.dof_pos[:,env.policy_dof_indices].cpu().numpy(),
                velocity=env.dof_vel[:,env.policy_dof_indices].cpu().numpy(),
                torque=env.torques[:,env.policy_dof_indices].cpu().numpy())
            # Position targets are constructed from the executed action, in policy order.
            target=env.default_dof_pos[:,env.policy_dof_indices].cpu().numpy().copy()
            target[:,[0,1,3,4]]+=data['executed'][:,[0,1,3,4]]*cfg.control.pos_action_scale
            target[:,[2,5]]=data['executed'][:,[2,5]]*cfg.control.vel_action_scale
            data['target']=target
            for key,value in data.items():joint_sums[key]+=np.nansum(value[active],axis=0)
            samples+=int(active.sum())
            signed_pitch+=sample['pitch'][active].sum();signed_roll+=sample['roll'][active].sum()
            if active.any() and step%10==0:
                row=dict(time_s=t+env.dt,pitch_deg=float(np.degrees(sample['pitch'][active].mean())),
                    roll_deg=float(np.degrees(sample['roll'][active].mean())),vx=float(sample['velocity'][active,0].mean()),
                    command_m_s=float(env.commands[active,0].mean()),
                    parking_valid_fraction=float(sample['parking_valid'][active].mean()),
                    vy=float(sample['velocity'][active,1].mean()),
                    yaw_rate=float(env.base_ang_vel[active,2].mean()),base_height=float(env.base_height[active].mean()),
                    parking_x=float((env.root_states[:,:2]-env.parking.reference)[active,0].mean()),
                    parking_y=float((env.root_states[:,:2]-env.parking.reference)[active,1].mean()))
                for key,value in data.items():
                    for i,name in enumerate(cfg.asset.policy_joint_names):row[key+'_'+name]=float(np.nanmean(value[active,i])) if np.isfinite(value[active,i]).any() else None
                if 'wheel_residual' in estimates:
                    record_wheel_residual(row,estimates['wheel_residual'],active,
                                          ppo.policy.wheel_residual_limit,cfg.normalization.obs_scales.lin_vel)
                trace.append(row)
                if env.odometry is not None:
                    sensor_delta=env.odometry.xy-env.odometry_reference
                    row['parking_sensor_x']=float(sensor_delta[active,0].mean())
                    row['parking_sensor_y']=float(sensor_delta[active,1].mean())
                if odometry is not None:
                    row['odometry_x']=float(odometry.xy[active,0].mean())
                    row['odometry_y']=float(odometry.xy[active,1].mean())
                    row['odometry_error_m']=float(odometry_error[active].mean())
            if camera is not None and step in capture_steps:
                from PIL import Image
                # GPU tensor state can advance while CPU graphics transforms
                # remain stale; synchronize before updating the camera scene.
                env.gym.fetch_results(env.sim,True)
                origin=env.root_states[0,:3].cpu().numpy()
                env.gym.set_camera_location(camera,env.envs[0],
                    gymapi.Vec3(float(origin[0]+1),float(origin[1]-1),float(origin[2]+.5)),
                    gymapi.Vec3(float(origin[0]),float(origin[1]),float(origin[2]-.1)))
                env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                pixels=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR)
                pixels=np.asarray(pixels,dtype=np.uint8).reshape(720,960,4)[:,:,:3]
                Image.fromarray(pixels).save(args.output_dir/('frame_{:05d}.png'.format(step)))
                evidence=dict(root_state=env.root_states[0].cpu().tolist(),
                    rigid_body_state=env.rigid_body_states[0].cpu().tolist(),
                    import_dof_names=env.dof_names,
                    policy_joint_names=cfg.asset.policy_joint_names)
                (args.output_dir/('state_{:05d}.json'.format(step))).write_text(json.dumps(evidence,indent=2))
            metrics.update(t+env.dt,sample)
        trials=metrics.rows(seed,args.scenario)
        if args.trace_failures:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            evidence=dict(checkpoint=str(args.checkpoint_path),seed=seed,dt=env.dt,
                observation_scales=config_dict(cfg.normalization.obs_scales),control=config_dict(cfg.control),
                policy_joint_order=cfg.asset.policy_joint_names,windows=failure_windows)
            (args.output_dir/'failure_windows.json').write_text(json.dumps(evidence))
        if args.scenario == 'tracking':
            for i,row in enumerate(trials):
                moving_count=phase_sums['forward']['count'][i]+phase_sums['reverse']['count'][i]
                moving_sq=phase_sums['forward']['sq'][i]+phase_sums['reverse']['sq'][i]
                row['steady_speed_rmse_m_s']=float(np.sqrt(moving_sq/moving_count)) if moving_count else float('nan')
                for phase,sums in phase_sums.items():
                    count=sums['count'][i]
                    row[phase+'_window_observed_s']=float(count*env.dt)
                    row[phase+'_speed_bias_m_s']=float(sums['bias'][i]/count) if count else float('nan')
                    row[phase+'_speed_rmse_m_s']=float(np.sqrt(sums['sq'][i]/count)) if count else float('nan')
        result=summarize(trials)[args.scenario]
        result.update(checkpoint=str(args.checkpoint_path),scenario=args.scenario,samples=samples,
            command_reference_limit=args.command_reference_limit,reference_action_limit=args.reference_action_limit,
            reference_ramp_rate=args.reference_ramp_rate,
            untrained_env_probe=args.spec is not None,
            untrained_policy_probe=args.policy_probe_spec is not None,policy_probe_overrides=policy_probe,
            untrained_reference_probe=args.command_reference_limit is not None,
            duration_s=args.duration,tracking_speed_m_s=args.tracking_speed if args.scenario=='tracking' else 0.,
            signed_pitch_deg=float(np.degrees(signed_pitch/max(samples,1))),
            signed_roll_deg=float(np.degrees(signed_roll/max(samples,1))),
            joint_means={k:(v/max(samples,1)).tolist() for k,v in joint_sums.items()},
            policy_joint_names=cfg.asset.policy_joint_names,
            target_units=['rad','rad','rad/s','rad','rad','rad/s'],
            action_std=model.action_std.detach().cpu().tolist())
        if odometry is not None:
            errors=odometry_error.cpu().numpy()[metrics.active]
            result['odometry_endpoint_error_mean_m']=float(errors.mean()) if len(errors) else None
            result['odometry_endpoint_error_p95_m']=float(np.quantile(errors,.95)) if len(errors) else None
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        write_csv(args.output_dir/'joint_trace.csv',trace)
        write_csv(args.output_dir/'trials.csv',trials)
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    torch.set_num_threads(1)
    args=arguments()
    train(args) if args.mode=='train' else diagnose(args)
