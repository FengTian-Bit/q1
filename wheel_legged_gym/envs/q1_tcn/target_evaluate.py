"""Full-range tracking evaluation: 90% accuracy, survival and parking regression."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path

import isaacgym  # before torch
import numpy as np
import torch
import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args, task_registry
from .actor_critic import ActorCriticTCN
from .evaluate import scenario_config, set_evaluation_commands
from .runner import load_checkpoint, config_dict


def command_at(time_s, vx, wz):
    direction=1. if 2<=time_s<6 else -1. if 9<=time_s<13 else 0.
    return direction*vx,direction*wz


def cases(stage, screen=False):
    values=[('linear060',.6,0.),('linear120',1.2,0.)] if screen else [
        ('linear030',.3,0.),('linear060',.6,0.),('linear090',.9,0.),('linear120',1.2,0.)]
    if stage=='turning':
        values += [('spin100',0.,1.),('combined120_100',1.2,1.),('combined120_m100',1.2,-1.)]
        if not screen:values += [('spin050',0.,.5),('combined060_050',.6,.5)]
    return values


def limits(vx, wz):
    return max(.03,.1*abs(vx)) if vx else .05, max(.05,.1*abs(wz)) if wz else .10


def number_mean(rows,key):
    values=[row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def evaluate_case(saved, checkpoint, args, name, vx, wz, seed, output):
    if output.exists():raise ValueError('Never overwrite a completed evaluation: {}'.format(output))
    duration=getattr(args,'stand_duration',20.) if name=='stand' else 16.
    cfg,train=scenario_config(saved['config'],'stand',seed,args.num_envs or 64,duration)
    args.seed=seed
    env,_=task_registry.make_env('q1_tcn',args=args,env_cfg=cfg)
    model=ActorCriticTCN(env.num_obs,env.num_privileged_obs,6,ablation=train.ablation,
                         **config_dict(train.policy)).to(args.rl_device)
    model.load_state_dict(saved['model_state_dict']);model.eval()
    count=env.num_envs;active=np.ones(count,dtype=bool);failure_time=np.full(count,np.nan)
    sums={phase:{key:np.zeros(count) for key in ['count','vx_sq','vx_bias','wz_sq','wz_bias']}
          for phase in ['forward','reverse','stop']}
    pitch_sq=np.zeros(count);roll_sq=np.zeros(count);observed=np.zeros(count);endpoint=np.zeros(count)
    command_acceleration_max=np.zeros(count)
    actual_acceleration_max=np.zeros(count)
    actual_acceleration_sq=np.zeros(count)
    trace=[]
    try:
        env.evaluation_mode=env.external_commands=True
        set_evaluation_commands(env,0.);env.reset()
        for step in range(round(duration/env.dt)):
            t=step*env.dt
            cv,cw=command_at(t,vx,wz)
            set_evaluation_commands(env,cv,cw)
            env.obs_buf[:,6:9]=env.commands[:,:3]*env.commands_scale
            env.obs_history[:,-env.num_obs+6:-env.num_obs+9]=env.obs_buf[:,6:9]
            with torch.no_grad():
                action,estimates=model.act_inference(env.obs_buf.to(args.rl_device),env.obs_history.to(args.rl_device))
                if step%10==0:
                    # Capture before step: automatic resets clear joint buffers.
                    wheel_velocity_pre=env.dof_vel[:,env.wheel_dof_indices].cpu().numpy().copy()
                    wheel_target=action[:,[2,5]].cpu().numpy()*cfg.control.vel_action_scale
                    wheel_logits=estimates['action_mean_unbounded'][:,[2,5]].cpu().numpy()
                    parking_weight=estimates.get('parking_weight',action.new_zeros(count)).cpu().numpy()
                    wheel_residual=estimates.get('wheel_residual',action.new_zeros((count,2))).cpu().numpy()
                    residual_activation=torch.maximum(env.obs_buf[:,6].abs(),
                        env.obs_buf[:,7].abs()*train.policy.wheel_residual_yaw_scale).clamp(max=1).cpu().numpy()
                    residual_bound=train.policy.wheel_residual_limit*residual_activation
                env.step(action.to(env.device))
            sample={key:value.cpu().numpy() for key,value in env.transition_metrics.items()}
            velocity=sample['velocity'][:,0];yaw=sample['yaw_rate']
            failed=active & sample['failure'].astype(bool)
            failure_time[failed]=t+env.dt
            observed[active]+=env.dt;pitch_sq[active]+=sample['pitch'][active]**2
            command_acceleration_max[active]=np.maximum(command_acceleration_max[active],np.abs(sample['command_acceleration'][active]))
            actual_acceleration_max[active]=np.maximum(actual_acceleration_max[active],np.abs(sample['linear_acceleration'][active]))
            actual_acceleration_sq[active]+=sample['linear_acceleration'][active]**2
            roll_sq[active]+=sample['roll'][active]**2;endpoint[active]=sample['parking_error'][active]
            phases=[]
            if name=='stand':
                if t>=duration-1:phases=['forward','stop']
            else:
                if 5<=t<6:phases.append('forward')
                if 12<=t<13:phases.append('reverse')
                if t>=15:phases.append('stop')
            for phase in phases:
                values=sums[phase];values['count'][active]+=1
                values['vx_sq'][active]+=(velocity[active]-cv)**2
                values['vx_bias'][active]+=velocity[active]-cv
                values['wz_sq'][active]+=(yaw[active]-cw)**2
                values['wz_bias'][active]+=yaw[active]-cw
            if step%10==0 and active.any():
                valid=active & sample['parking_valid'].astype(bool)
                row=dict(time_s=t+env.dt,target_vx=cv,command_vx=float(sample['command'][active].mean()),command_wz=cw,
                    vx=float(velocity[active].mean()),wz=float(yaw[active].mean()),
                    command_acceleration=float(sample['command_acceleration'][active].mean()),
                    actual_acceleration=float(sample['linear_acceleration'][active].mean()),
                    actual_acceleration_min=float(sample['linear_acceleration'][active].min()),
                    actual_acceleration_max=float(sample['linear_acceleration'][active].max()),
                    pitch_deg=float(np.degrees(sample['pitch'][active].mean())),
                    roll_deg=float(np.degrees(sample['roll'][active].mean())),
                    parking_error_m=float(sample['parking_error'][valid].mean()) if valid.any() else None,
                    alive=int(active.sum()),parking_weight=float(parking_weight[active].mean()))
                for side,index in [('left',0),('right',1)]:
                    row['wheel_target_'+side]=float(wheel_target[active,index].mean())
                    row['wheel_velocity_pre_'+side]=float(wheel_velocity_pre[active,index].mean())
                    row['wheel_logit_'+side]=float(wheel_logits[active,index].mean())
                    row['wheel_residual_'+side]=float(wheel_residual[active,index].mean())
                    row['wheel_residual_near_limit_'+side]=float(((residual_bound[active]>1e-6)&
                        (np.abs(wheel_residual[active,index])>=.95*residual_bound[active])).mean())
                trace.append(row)
            active[failed]=False
            if not active.any():break
        rows=[]
        for i in range(count):
            row=dict(trial=i,seed=seed,failed=bool(np.isfinite(failure_time[i])),
                     failure_time_s=float(failure_time[i]) if np.isfinite(failure_time[i]) else None,
                     observed_s=float(observed[i]),parking_end_m=float(endpoint[i]),
                     command_acceleration_max=float(command_acceleration_max[i]),
                     actual_acceleration_max=float(actual_acceleration_max[i]),
                     actual_acceleration_rms=math.sqrt(actual_acceleration_sq[i]/max(1,observed[i]/env.dt)),
                     pitch_rms_deg=math.degrees(math.sqrt(pitch_sq[i]/max(1,observed[i]/env.dt))),
                     roll_rms_deg=math.degrees(math.sqrt(roll_sq[i]/max(1,observed[i]/env.dt))))
            for phase,values in sums.items():
                n=values['count'][i];row[phase+'_count']=int(n)
                for axis in ['vx','wz']:
                    row[phase+'_'+axis+'_rmse']=math.sqrt(values[axis+'_sq'][i]/n) if n else None
                    row[phase+'_'+axis+'_bias']=float(values[axis+'_bias'][i]/n) if n else None
            rows.append(row)
        lv,lw=limits(vx,wz)
        checks=dict(survival=all(not row['failed'] for row in rows),
                    full_horizon=all(row['observed_s']>=duration-.02 for row in rows))
        if cfg.commands.max_linear_acceleration>0:
            checks['command_acceleration']=bool(command_acceleration_max.max()<=cfg.commands.max_linear_acceleration+1e-4)
        moving=['forward'] if name=='stand' else ['forward','reverse']
        for phase in moving:
            checks[phase+'_complete']=all(row[phase+'_count']>=round(1/env.dt)-1 for row in rows)
            for axis,limit in [('vx',lv),('wz',lw)]:
                for metric in ['rmse','bias']:
                    key=phase+'_'+axis+'_'+metric;value=number_mean(rows,key)
                    checks[key]=value is not None and abs(value)<=limit
        checks['stop_vx']=number_mean(rows,'stop_vx_rmse') is not None and number_mean(rows,'stop_vx_rmse')<=.08
        checks['stop_wz']=number_mean(rows,'stop_wz_rmse') is not None and number_mean(rows,'stop_wz_rmse')<=.15
        checks['parking_end']=number_mean(rows,'parking_end_m')<=(.025 if name=='stand' else .05)
        checks['parking_p95']=float(np.quantile([row['parking_end_m'] for row in rows],.95))<=(.05 if name=='stand' else .10)
        checks['posture']=max(number_mean(rows,'pitch_rms_deg'),number_mean(rows,'roll_rms_deg'))<=(4. if name=='stand' else 10.)
        result=dict(case=name,stage=args.stage,seed=seed,vx_amplitude=vx,wz_amplitude=wz,
            trials=count,survived=sum(not row['failed'] for row in rows),duration_s=duration,
            passed=all(checks.values()),checks=checks,
            metrics={key:number_mean(rows,key) for key in rows[0] if key not in ['failed','trial','seed','failure_time_s']},
            parking_p95_m=float(np.quantile([row['parking_end_m'] for row in rows],.95)),
            tolerance_vx=lv,tolerance_wz=lw,checkpoint=str(checkpoint.resolve()))
        for axis,amplitude in [('vx',vx),('wz',wz)]:
            if amplitude:
                errors=[result['metrics'][phase+'_'+axis+'_rmse'] for phase in moving]
                result[axis+'_tracking_score']=max(0.,1-float(np.mean(errors))/abs(amplitude)) if all(value is not None for value in errors) else 0.
        output.mkdir(parents=True)
        for filename,values in [('trials.csv',rows),('trace.csv',trace)]:
            with (output/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        (output/'result.json').write_text(json.dumps(result,indent=2))
        plot_case(output,trace,name,cfg.commands.max_linear_acceleration,(result['survived'],count))
        return result
    finally:
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
        del model,env
        gc.collect();torch.cuda.empty_cache()


def plot_case(output, rows, title, acceleration_limit=0., survival=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    t=[row['time_s'] for row in rows]
    for ax,actual,target,label in [(axes[0],'vx','command_vx','m/s'),(axes[1],'wz','command_wz','rad/s')]:
        ax.plot(t,[row[actual] for row in rows],label=actual)
        ax.plot(t,[row[target] for row in rows],'--',label='command');ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
    if any(abs(row['target_vx']-row['command_vx'])>1e-5 for row in rows):
        axes[0].plot(t,[row['target_vx'] for row in rows],':',color='grey',label='target')
        axes[0].legend()
    for key in ['pitch_deg','roll_deg']:axes[2].plot(t,[row[key] for row in rows],label=key)
    axes[2].set_ylabel('deg');axes[2].legend();axes[2].grid(alpha=.25)
    for key in ['command_acceleration','actual_acceleration']:
        axes[3].plot(t,[row[key] for row in rows],label=key)
    if 'actual_acceleration_min' in rows[0]:
        axes[3].fill_between(t,[row['actual_acceleration_min'] for row in rows],
                            [row['actual_acceleration_max'] for row in rows],
                            color='tab:orange',alpha=.15,label='actual trial range')
    if acceleration_limit>0:
        axes[3].axhline(acceleration_limit,color='grey',linestyle='--',label='command limit')
        axes[3].axhline(-acceleration_limit,color='grey',linestyle='--')
    axes[3].set_ylabel('m/s^2');axes[3].set_xlabel('Time (s)');axes[3].legend();axes[3].grid(alpha=.25)
    if survival is not None:
        title+=' | survived {}/{} | curves: active trial means'.format(*survival)
    fig.suptitle(title);fig.savefig(output/'response.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint_path',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['linear','turning'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1201,1202,1203])
    parser.add_argument('--screen',action='store_true')
    parser.add_argument('--stand_duration',type=float,default=None)
    opts,remaining=parser.parse_known_args()
    import sys
    sys.argv=[sys.argv[0]]+remaining;args=get_args(default_task='q1_tcn')
    for key,value in vars(opts).items():setattr(args,key,value)
    if args.stand_duration is None:
        args.stand_duration=20. if args.screen else 60.
    if args.stand_duration<2:
        raise ValueError('stand_duration must include at least two seconds')
    torch.set_num_threads(1)
    if args.output_dir.exists():raise ValueError('Use a new evaluation folder')
    args.output_dir.mkdir(parents=True)
    saved=load_checkpoint(args.checkpoint_path);results=[]
    ranges=saved['config']['env']['commands']['ranges']
    if min(-ranges['lin_vel_x'][0],ranges['lin_vel_x'][1])<1.2:raise ValueError('Checkpoint was not configured for +/-1.2')
    if args.stage=='turning' and min(-ranges['ang_vel_yaw'][0],ranges['ang_vel_yaw'][1])<1:raise ValueError('Checkpoint was not configured for +/-1 yaw')
    for seed in args.seeds:
        for name,vx,wz in cases(args.stage,args.screen):
            result=evaluate_case(saved,args.checkpoint_path,args,name,vx,wz,seed,args.output_dir/('{}_seed{}'.format(name,seed)))
            results.append(result);print(name,seed,'passed',result['passed'],flush=True)
    seed=args.seeds[0]
    results.append(evaluate_case(saved,args.checkpoint_path,args,'stand',0.,0.,seed,args.output_dir/('stand_seed{}'.format(seed))))
    verdict=dict(stage=args.stage,checkpoint=str(args.checkpoint_path.resolve()),
        model_sha256=hashlib.sha256(args.checkpoint_path.read_bytes()).hexdigest(),seeds=args.seeds,
        accuracy_goal=.90,passed=all(row['passed'] for row in results),cases=results,
        stand_duration_s=args.stand_duration,
        max_linear_command_acceleration=saved['config']['env']['commands'].get('max_linear_acceleration',0.),
        reward_version=saved['config']['env']['rewards']['reward_version'],
        relative_tracking=saved['config']['env']['rewards'].get('relative_tracking',{}),
        reward_scales=saved['config']['env']['rewards']['scales'],
        wheel_diff_tracking=saved['config']['env']['rewards'].get('wheel_diff_tracking',{}),
        note='Scores are 1 - steady per-trial RMSE / command magnitude; zero axes use absolute tolerance. Steady windows5-6s and12-13s; stop15-16s; original strict2.5cm/s protocol is not used.')
    (args.output_dir/'summary.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps({key:value for key,value in verdict.items() if key!='cases'},indent=2))


if __name__=='__main__':main()
