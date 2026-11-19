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
from .q1_tcn import SLOPE_LABEL_SCALE, CLEARANCE_LABEL_SCALE
from .runner import load_checkpoint, config_dict


def command_at(time_s, vx, wz):
    direction=1. if 2<=time_s<6 else -1. if 9<=time_s<13 else 0.
    return direction*vx,direction*wz


def cases(stage, screen=False):
    if stage=='stand':
        return []
    if stage=='slope':
        return [(name,vx,0.) for name,vx,_ in SLOPE_CASES]
    if stage=='stairs':
        return [(name,vx,0.) for name,vx,_ in STAIR_CASES]
    values=[('linear060',.6,0.),('linear120',1.2,0.)] if screen else [
        ('linear030',.3,0.),('linear060',.6,0.),('linear090',.9,0.),('linear120',1.2,0.)]
    if stage=='turning':
        values += [('spin100',0.,1.),('combined120_100',1.2,1.),('combined120_m100',1.2,-1.)]
        if not screen:values += [('spin050',0.,.5),('combined060_050',.6,.5)]
    return values


# Stage 1 slope protocol (docs/q1_terrain_stage01.md 2.7.1). Each case runs on one
# T0 tile. "slope_down" tiles are pits with a flat 1 m centre platform: a robot
# starting on the platform edge (x=0.4) facing +x drives uphill when moving forward
# and downhill when reversing. Robots start on flat ground as in training and drive
# onto the slope; spawning at rest on a 15 deg face is never seen in training.
#   kind 'hold':   drive up at 0.3 m/s to x=2 m, then hold zero command for 20 s
#   kind 'cross':  as hold, but turn 90 deg in place (0.5 rad/s) before holding
#   kind 'updown': 16 s timeline, windows 4-5 s (uphill) and 10.5-11.5 s (downhill)
#   name: (case, vx amplitude, terrain dict)
SLOPE_START_X, SLOPE_HOLD_X, SLOPE_DRIVE, SLOPE_HOLD_S, SLOPE_TURN = .4, 2., .3, 20., .5
SLOPE_CASES = [
    ('stand_slope00', 0., dict(tile='flat', degrees=0., kind='hold')),
    ('stand_slope05', 0., dict(tile='slope_down', degrees=5., kind='hold')),
    ('stand_slope10', 0., dict(tile='slope_down', degrees=10., kind='hold')),
    ('stand_slope15', 0., dict(tile='slope_down', degrees=15., kind='hold')),
    ('updown15_030', .3, dict(tile='slope_down', degrees=15., kind='updown')),
    ('updown15_060', .6, dict(tile='slope_down', degrees=15., kind='updown')),
    ('cross_slope15', 0., dict(tile='slope_down', degrees=15., kind='cross')),
]
SLOPE_TERRAIN = {name: terrain for name,_,terrain in SLOPE_CASES}

# Stage 2 stair protocol. T0 naming: a centre-spawn robot drives UP on the
# inverted "stairs_down" tile and DOWN on "stairs_up". Every robot starts on the
# 1 m platform edge (x=0.4) facing +x, drives forward at STAIR_DRIVE for
# STAIR_DRIVE_S seconds (about 2.4 m, several 0.31 m treads), then holds still.
# The case overrides the tile list to that single tile at one riser height, so
# checkpoints trained without stairs can be scored too.
# Stair success (tools/q1_stairs_summary.py): survived and travelled >= 1.5 m.
STAIR_DRIVE, STAIR_DRIVE_S, STAIR_HOLD_S = .3, 8., 3.
STAIR_CASES = [('stairs_{}_h{:02d}'.format(direction, round(100*height)), STAIR_DRIVE,
                dict(tile='stairs_down' if direction=='up' else 'stairs_up', height=height, kind='drive'))
               for height in (.04,.08,.12,.16,.20) for direction in ('up','down')]
SLOPE_TERRAIN.update({name: terrain for name,_,terrain in STAIR_CASES})
CALF_LARGE_RAD = 0.81


def slope_schedule(terrain, vx):
    """(duration, command(t) -> (vx, wz), hold start time or None) for one slope case."""
    if terrain['kind']=='drive':
        return (1.+STAIR_DRIVE_S+STAIR_HOLD_S,
                lambda t: (vx,0.) if 1.<=t<1.+STAIR_DRIVE_S else (0.,0.), None)
    if terrain['kind']=='updown':
        return 16., lambda t: command_at(t,vx,0.), None
    drive_end=1.+(SLOPE_HOLD_X-SLOPE_START_X)/SLOPE_DRIVE
    turn_end=drive_end+(math.pi/2/SLOPE_TURN if terrain['kind']=='cross' else 0.)
    def command(t):
        if 1.<=t<drive_end:return SLOPE_DRIVE,0.
        if drive_end<=t<turn_end:return 0.,SLOPE_TURN
        return 0.,0.
    return turn_end+SLOPE_HOLD_S, command, turn_end


def slope_phases(terrain, t, duration, hold_start):
    if terrain['kind']=='drive':
        return ((['forward'] if 3.<=t<1.+STAIR_DRIVE_S else [])
                +(['stop'] if t>=duration-1 else []))
    if terrain['kind']=='updown':
        return ((['forward'] if 4<=t<5 else [])+(['reverse'] if 10.5<=t<11.5 else [])
                +(['stop'] if t>=15 else []))
    return ['forward','stop'] if t>=duration-1 else []


def apply_slope_terrain(cfg, terrain):
    """One selected T0 tile; every robot of the case starts at the same pose."""
    cfg.terrain.mesh_type='trimesh'
    cfg.terrain.curriculum=False
    cfg.terrain.num_rows=cfg.terrain.num_cols=1
    cfg.terrain.selected=True
    cfg.terrain.selected_type=terrain['tile']
    # strict_v2: the protocol geometry (start x=0.4 on a 1 m platform) must not
    # follow the training platform_size, or slope holds stop on flat ground.
    cfg.terrain.platform_size=float(__import__('os').environ.get('Q1_EVAL_PLATFORM',1.0))
    if 'height' in terrain:
        from wheel_legged_gym.envs.q1.t0_terrain import Q1T0Terrain
        cfg.terrain.terrain_names=[terrain['tile']]
        cfg.terrain.terrain_labels=[Q1T0Terrain.TYPE_LABELS[terrain['tile']]]
        cfg.terrain.terrain_proportions=[1.0]
        cfg.terrain.stair_height_range=[terrain['height'],terrain['height']]
        cfg.terrain.stair_width=float(__import__("os").environ.get("Q1_STAIR_WIDTH",0.50))  # default 0.50; training used 0.31
        cfg.terrain.selected_difficulty=0.
        return
    rise=math.tan(math.radians(terrain['degrees']))
    low,high=cfg.terrain.slope_range
    cfg.terrain.selected_difficulty=0. if terrain['tile']=='flat' else min(1.,max(0.,(rise-low)/(high-low)))


def place_on_slope(env, terrain):
    """Start every robot on the flat platform edge, facing +x (towards the slope)."""
    offset=env.env_origins.new_tensor([SLOPE_START_X if terrain['tile']!='flat' else 0.,0.])
    env.env_origins[:,:2]+=offset
    env.env_origins[:,2]=env._terrain_height_at(env.env_origins[:,:2])


def limits(vx, wz):
    return max(.03,.1*abs(vx)) if vx else .05, max(.05,.1*abs(wz)) if wz else .10


def number_mean(rows,key):
    values=[row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def evaluate_case(saved, checkpoint, args, name, vx, wz, seed, output):
    if output.exists():raise ValueError('Never overwrite a completed evaluation: {}'.format(output))
    terrain=SLOPE_TERRAIN.get(name)
    # Slope cases without a speed command reuse the standing windows and limits.
    standing=name=='stand' or (terrain is not None and vx==0 and wz==0)
    hold_start=None
    if terrain is not None:
        duration,command,hold_start=slope_schedule(terrain,vx)
    else:
        duration=getattr(args,'stand_duration',20.) if name=='stand' else 16.
        command=lambda t:command_at(t,vx,wz)
    cfg,train=scenario_config(saved['config'],'stand',seed,args.num_envs or 64,duration)
    if terrain is not None:
        apply_slope_terrain(cfg,terrain)
    args.seed=seed
    env,_=task_registry.make_env('q1_tcn',args=args,env_cfg=cfg)
    if terrain is not None:
        place_on_slope(env,terrain)
    model=ActorCriticTCN(env.num_obs,env.num_privileged_obs,6,ablation=train.ablation,
                         **config_dict(train.policy)).to(args.rl_device)
    model.load_state_dict(saved['model_state_dict']);model.eval()
    count=env.num_envs;active=np.ones(count,dtype=bool);failure_time=np.full(count,np.nan)
    sums={phase:{key:np.zeros(count) for key in ['count','vx_sq','vx_bias','wz_sq','wz_bias']}
          for phase in ['forward','reverse','stop']}
    pitch_sq=np.zeros(count);roll_sq=np.zeros(count);observed=np.zeros(count);endpoint=np.zeros(count)
    # user_v4 posture: steady forward/reverse windows versus all other (acceleration) time.
    posture={part:{key:np.zeros(count) for key in ["n","pitch_sq","roll_sq"]} for part in ["steady","transient","hold"]}
    # Stage 0 range use (S0-8/9) and base height samples (S0-10). The doc's
    # |a_calf| > 0.9 at calf scale 0.9 is a knee offset of 0.81 rad.
    calf_large=np.zeros(count);heights=[]
    # World base height, for the climb criterion. base_height is relative to the
    # terrain sampled around the base (0.5 m ahead included), so it cannot show one.
    base_z_series=[]
    # CTBC diagnostics: did a riser collision trigger a swing, and how high did the wheel go.
    trigger_count=np.zeros(count);swing_s=np.zeros(count);swing_lift=np.zeros(count)
    joint_scales=getattr(cfg.control,'leg_action_scales',None)
    calf_scale=joint_scales[1] if joint_scales is not None else cfg.control.pos_action_scale
    # Stage 1 terrain-head errors (T1-7 slope angle, T1-8 clearance) and lateral drift (T1-6).
    head_sq={key:np.zeros(count) for key in ['slope_deg','clearance_cm']};head_n=np.zeros(count)
    start_xy=None;lateral_drift=np.zeros(count)
    command_acceleration_max=np.zeros(count)
    actual_acceleration_max=np.zeros(count)
    actual_acceleration_sq=np.zeros(count)
    trace=[]
    ground_start=x_start=z_settled=None
    ground_end=np.full(count,np.nan);x_end=np.full(count,np.nan);z_end=np.full(count,np.nan)
    try:
        env.evaluation_mode=env.external_commands=True
        set_evaluation_commands(env,0.);env.reset()
        for step in range(round(duration/env.dt)):
            t=step*env.dt
            cv,cw=command(t)
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
            calf_large[active]+=(np.abs(sample['action'][active][:,[1,4]])*calf_scale>CALF_LARGE_RAD).any(axis=1)
            heights.append(np.where(active,sample['base_height'],np.nan))
            base_z_series.append(np.where(active,sample['position'][:,2],np.nan))
            # strict_v2: world ground under the wheels, start vs last alive sample.
            ground_now=env._terrain_height_at(env.rigid_body_states[:,env.feet_indices,:2]).mean(dim=1).cpu().numpy()
            if ground_start is None:
                ground_start=ground_now.copy();x_start=sample['position'][:,0].copy()
            if z_settled is None and t>=1.-1e-9:z_settled=sample['position'][:,2].copy()
            ground_end[active]=ground_now[active];x_end[active]=sample['position'][active,0]
            z_end[active]=sample['position'][active,2]
            if hasattr(env,'guidance'):
                trig=env.guidance.triggered.any(dim=1).cpu().numpy()
                swing=env.guidance.swing_mask
                lift=((env.ctbc_wheel_bottom()-env.guidance.start_height)*swing).max(dim=1).values.cpu().numpy()
                trigger_count[active]+=trig[active]
                swing_s[active]+=swing.any(dim=1).cpu().numpy()[active]*env.dt
                swing_lift[active]=np.maximum(swing_lift[active],lift[active])
            position=sample['position'][:,:2]
            if start_xy is None:start_xy=position.copy()
            lateral_drift[active]=np.maximum(lateral_drift[active],
                np.linalg.norm(position[active]-start_xy[active],axis=1))
            if 'slope' in estimates or 'clearance' in estimates:
                head_n[active]+=1
                target=sample['targets']
                if 'slope' in estimates:
                    est=estimates['slope'].cpu().numpy()/SLOPE_LABEL_SCALE
                    true=target[:,10:12]/SLOPE_LABEL_SCALE
                    normals=[np.concatenate((xy,np.sqrt(np.clip(1-(xy**2).sum(1,keepdims=True),0,1))),1)
                             for xy in (np.clip(est,-1,1),true)]
                    cosine=np.clip((normals[0]*normals[1]).sum(1),-1,1)
                    head_sq['slope_deg'][active]+=np.degrees(np.arccos(cosine))[active]**2
                if 'clearance' in estimates:
                    error=100*(estimates['clearance'].cpu().numpy()-target[:,12:14])/CLEARANCE_LABEL_SCALE
                    head_sq['clearance_cm'][active]+=(error**2).mean(1)[active]
            if terrain is not None:
                phases=slope_phases(terrain,t,duration,hold_start)
            elif standing:
                phases=['forward','stop'] if t>=duration-1 else []
            else:
                phases=[]
                if 5<=t<6:phases.append('forward')
                if 12<=t<13:phases.append('reverse')
                if t>=15:phases.append('stop')
            if hold_start is not None and t>=hold_start:
                part="hold"  # slope standing posture only after the robot has stopped on the face
            else:
                part="steady" if not standing and ("forward" in phases or "reverse" in phases) else "transient"
            posture[part]["n"][active]+=1;posture[part]["pitch_sq"][active]+=sample["pitch"][active]**2
            posture[part]["roll_sq"][active]+=sample["roll"][active]**2
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
        heights=np.array(heights) if heights else np.full((1,count),np.nan)
        base_z=np.array(base_z_series) if base_z_series else np.full((1,count),np.nan)
        start_z=np.full(count,np.nan);max_z=np.full(count,np.nan);min_z=np.full(count,np.nan)
        for i in range(count):
            series=base_z[:,i];series=series[np.isfinite(series)]
            if len(series):start_z[i]=series[0];max_z[i]=series.max();min_z[i]=series.min()
        for i in range(count):
            frames=max(1,observed[i]/env.dt)
            row=dict(trial=i,seed=seed,failed=bool(np.isfinite(failure_time[i])),
                     failure_time_s=float(failure_time[i]) if np.isfinite(failure_time[i]) else None,
                     observed_s=float(observed[i]),parking_end_m=float(endpoint[i]),
                     command_acceleration_max=float(command_acceleration_max[i]),
                     actual_acceleration_max=float(actual_acceleration_max[i]),
                     actual_acceleration_rms=math.sqrt(actual_acceleration_sq[i]/frames),
                     pitch_rms_deg=math.degrees(math.sqrt(pitch_sq[i]/frames)),
                     roll_rms_deg=math.degrees(math.sqrt(roll_sq[i]/frames)),
                     calf_large_fraction=float(calf_large[i]/frames),
                     base_height_p05=float(np.nanquantile(heights[:,i],.05)) if np.isfinite(heights[:,i]).any() else None,
                     max_displacement_m=float(lateral_drift[i]),
                     start_base_z=float(start_z[i]) if np.isfinite(start_z[i]) else None,
                     max_base_z=float(max_z[i]) if np.isfinite(max_z[i]) else None,
                     climb_m=float(max_z[i]-start_z[i]) if np.isfinite(max_z[i]) and np.isfinite(start_z[i]) else None,
                     min_base_z=float(min_z[i]) if np.isfinite(min_z[i]) else None,
                     drop_m=float(start_z[i]-min_z[i]) if np.isfinite(min_z[i]) and np.isfinite(start_z[i]) else None,
                     trigger_count=int(trigger_count[i]),swing_s=float(swing_s[i]),
                     max_swing_lift_m=float(swing_lift[i]),
                     ground_rise_m=float(ground_end[i]-ground_start[i]) if ground_start is not None and np.isfinite(ground_end[i]) else None,
                     final_dx_m=float(x_end[i]-x_start[i]) if x_start is not None and np.isfinite(x_end[i]) else None,
                     base_rise_settled_m=float(z_end[i]-z_settled[i]) if z_settled is not None and np.isfinite(z_end[i]) else None,
                     platform_size_m=float(cfg.terrain.platform_size))
            for key,values in head_sq.items():
                row[key+'_rmse']=math.sqrt(values[i]/head_n[i]) if head_n[i] else None
            for part,values in posture.items():
                n=values["n"][i];row[part+"_posture_count"]=int(n)
                for axis in ["pitch","roll"]:
                    row[part+"_"+axis+"_rms_deg"]=math.degrees(math.sqrt(values[axis+"_sq"][i]/n)) if n else None
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
        moving=['forward'] if standing else ['forward','reverse']
        for phase in moving:
            checks[phase+'_complete']=all(row[phase+'_count']>=round(1/env.dt)-1 for row in rows)
            for axis,limit in [('vx',lv),('wz',lw)]:
                for metric in ['rmse','bias']:
                    key=phase+'_'+axis+'_'+metric;value=number_mean(rows,key)
                    checks[key]=value is not None and abs(value)<=limit
        checks['stop_vx']=number_mean(rows,'stop_vx_rmse') is not None and number_mean(rows,'stop_vx_rmse')<=.08
        checks['stop_wz']=number_mean(rows,'stop_wz_rmse') is not None and number_mean(rows,'stop_wz_rmse')<=.15
        checks['parking_end']=number_mean(rows,'parking_end_m')<=(.025 if standing else .05)
        checks['parking_p95']=float(np.quantile([row['parking_end_m'] for row in rows],.95))<=(.05 if standing else .10)
        checks['posture']=max(number_mean(rows,'pitch_rms_deg'),number_mean(rows,'roll_rms_deg'))<=(4. if standing else 10.)
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
    parser.add_argument('--stage',choices=['stand','linear','turning','slope','stairs'],required=True)
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
    if args.stage!='stand' and min(-ranges['lin_vel_x'][0],ranges['lin_vel_x'][1])<1.2:raise ValueError('Checkpoint was not configured for +/-1.2')
    if args.stage=='turning' and min(-ranges['ang_vel_yaw'][0],ranges['ang_vel_yaw'][1])<1:raise ValueError('Checkpoint was not configured for +/-1 yaw')
    for seed in args.seeds:
        for name,vx,wz in cases(args.stage,args.screen):
            result=evaluate_case(saved,args.checkpoint_path,args,name,vx,wz,seed,args.output_dir/('{}_seed{}'.format(name,seed)))
            results.append(result);print(name,seed,'passed',result['passed'],flush=True)
    stand_seeds=[] if args.stage in ('slope','stairs') else args.seeds if args.stage=='stand' else args.seeds[:1]
    for seed in stand_seeds:
        results.append(evaluate_case(saved,args.checkpoint_path,args,'stand',0.,0.,seed,args.output_dir/('stand_seed{}'.format(seed))))
    verdict=dict(stage=args.stage,checkpoint=str(args.checkpoint_path.resolve()),
        model_sha256=hashlib.sha256(args.checkpoint_path.read_bytes()).hexdigest(),seeds=args.seeds,
        accuracy_goal=None if args.stage=='stand' else .90,passed=all(row['passed'] for row in results),cases=results,
        stand_duration_s=args.stand_duration,
        max_linear_command_acceleration=saved['config']['env']['commands'].get('max_linear_acceleration',0.),
        reward_version=saved['config']['env']['rewards']['reward_version'],
        relative_tracking=saved['config']['env']['rewards'].get('relative_tracking',{}),
        multi_scale_tracking=saved['config']['env']['rewards'].get('multi_scale_tracking',{}),
        multi_scale_parking=saved['config']['env']['rewards'].get('multi_scale_parking',{}),
        adaptive_tracking_sigma=saved['config']['env']['rewards'].get('adaptive_tracking_sigma',{}),
        yaw_tracking_mixture=saved['config']['env']['rewards'].get('yaw_tracking_mixture',{}),
        adaptive_yaw_tracking_sigma=saved['config']['env']['rewards'].get('adaptive_yaw_tracking_sigma',{}),
        positive_stand_rewards=saved['config']['env']['rewards'].get('positive_stand_rewards',{}),
        reward_scales=saved['config']['env']['rewards']['scales'],
        wheel_diff_tracking=saved['config']['env']['rewards'].get('wheel_diff_tracking',{}),
        note=('Standing only: survival, posture, drift and final-second velocity; no speed/turning qualification.' if args.stage=='stand' else
              'Scores are 1 - steady per-trial RMSE / command magnitude; zero axes use absolute tolerance. Steady windows5-6s and12-13s; stop15-16s; original strict2.5cm/s protocol is not used.'))
    (args.output_dir/'summary.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps({key:value for key,value in verdict.items() if key!='cases'},indent=2))


if __name__=='__main__':main()
