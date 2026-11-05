"""Fixed scenarios, independent evaluation seeds, CSV/JSON and response plots."""

import csv
import json
from pathlib import Path

import isaacgym  # Isaac Gym must be imported before torch.
import numpy as np
import torch

from wheel_legged_gym.utils import get_args, task_registry
from .q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO, validate_config
from .actor_critic import ActorCriticTCN
from .runner import restore_config, config_dict, load_checkpoint
from .metrics import TrialMetrics, summarize
from .cli import get_tcn_args, SCENARIOS


def write_csv(path, rows):
    names = list(dict.fromkeys(name for row in rows for name in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def command_at(time_s, scenario):
    if scenario == "stand" or scenario.startswith("push_"):
        return 0.0
    # Identical forward / stop / reverse sequence; within training command range.
    return 0.0 if time_s < 2 else 0.25 if time_s < 5 else 0.0 if time_s < 7 else -0.25 if time_s < 10 else 0.0


def evaluation_height(cfg):
    height = float(cfg.rewards.base_height_target)
    lower, upper = cfg.commands.ranges.height
    if not np.isfinite(height) or not lower <= height <= upper or height <= 0:
        raise ValueError("Evaluation base_height_target must be positive and inside the trained height command range")
    return height


def set_evaluation_commands(env, forward_speed, yaw_rate=0.0):
    """Set all three commands, including height, before reset fills history."""
    env.commands[:, 2] = evaluation_height(env.cfg)
    env.set_command_targets(forward_speed, yaw_rate)


def scenario_config(saved, scenario, seed, num_envs, duration):
    cfg, train_cfg = Q1TCNCfg(), Q1TCNCfgPPO()
    # Old checkpoints retain their original reward, despite newer class defaults.
    cfg.rewards.reward_version = saved["env"]["rewards"].get("reward_version", 1)
    if cfg.rewards.reward_version == 1:
        cfg.rewards.scales.parking_drift = 0.0
    restore_config(cfg, saved["env"])
    restore_config(train_cfg, saved["train"])
    cfg.seed, cfg.env.num_envs = seed, num_envs
    cfg.env.episode_length_s = duration + 2.0  # evaluation horizon never triggers a reset
    cfg.commands.resampling_time = duration + 10.0
    cfg.commands.sequence_fraction = 0.0  # evaluation commands always come from its protocol
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_inertia = False
    cfg.domain_rand.randomize_Kp = False
    cfg.domain_rand.randomize_Kd = False
    cfg.domain_rand.randomize_motor_torque = False
    cfg.domain_rand.randomize_default_dof_pos = False
    cfg.domain_rand.randomize_action_delay = False
    # Keep calls to _push_robots enabled so evaluation force is applied each step.
    cfg.domain_rand.push_robots = True
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = 0.2 if scenario == "low_friction" else 0.8
    if scenario == "payload":
        cfg.domain_rand.randomize_base_mass = True
        cfg.domain_rand.added_mass_range = [0.75, 0.75]
        cfg.estimator.payload_com_offset_x = 0.02
    validate_config(cfg, train_cfg)
    return cfg, train_cfg


def plot_trace(path, traces, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for seed, rows in traces:
        t = np.array([row["time_s"] for row in rows])
        value = lambda name: np.array([row[name] for row in rows])
        axes[0, 0].plot(t, value("velocity_m_s"), label="seed {}".format(seed))
        axes[0, 0].fill_between(t, value("velocity_m_s")-value("velocity_std_m_s"),
                              value("velocity_m_s")+value("velocity_std_m_s"), alpha=0.1)
        axes[0, 1].plot(t, value("pitch_rad"))
        axes[1, 0].plot(t, value("roll_rad"))
        axes[1, 1].plot(t, value("slip_abs_m_s"))
        axes[2, 0].plot(t, value("force_n"))
        axes[2, 1].plot(t, value("alive_count"))
        axes[3, 0].plot(t, value("parking_error_m"))
        axes[3, 1].plot(t, value("action_clipping_fraction"))
    if traces:
        rows = traces[0][1]
        axes[0, 0].plot([row["time_s"] for row in rows], [row["command_m_s"] for row in rows],
                       "k--", label="command")
    for axis, ylabel in zip(axes.flat, ("vx [m/s]", "pitch [rad]", "roll [rad]",
                                        "supported slip [m/s]", "wheel Fz total [N]", "observed trials",
                                        "locked parking error [m]", "action clipping fraction")):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.set_xlabel("time [s]")
    axes[0, 0].legend()
    fig.suptitle(title + " | curves include trials alive at each step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = get_tcn_args(get_args, evaluation=True)
    if args.task != "q1_tcn" or args.duration < 7.0 or args.push_force <= 0:
        raise ValueError("Use task=q1_tcn, duration >= 7 s and positive push_force")
    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    saved = checkpoint["config"]
    if saved.get("task") != "q1_tcn" or saved.get("schema_version") != 1:
        raise ValueError("Expected a q1_tcn schema_version=1 checkpoint")
    num_envs = args.num_envs if args.num_envs is not None else 64
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    output = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / ("eval_"+checkpoint_path.stem)
    output.mkdir(parents=True, exist_ok=True)
    rows, scenario_traces = [], {}
    for scenario in args.scenarios:
        scenario_traces[scenario] = []
        for seed in args.eval_seeds:
            cfg, train_cfg = scenario_config(saved, scenario, seed, num_envs, args.duration)
            args.seed = seed
            env, _ = task_registry.make_env("q1_tcn", args=args, env_cfg=cfg)
            try:
                model = ActorCriticTCN(env.num_obs, env.num_privileged_obs, env.num_actions,
                    ablation=train_cfg.ablation, **config_dict(train_cfg.policy)).to(args.rl_device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                env.evaluation_mode, env.external_commands = True, True
                set_evaluation_commands(env, 0.0)
                env.reset()
                push_end = 3.15 if scenario.startswith("push_") else None
                metrics = TrialMetrics(num_envs, env.dt, env.root_states[:, :3].cpu().numpy(),
                                       model.enabled, push_end)
                for step in range(round(args.duration/env.dt)):
                    time_s = step*env.dt
                    set_evaluation_commands(env, command_at(time_s, scenario))
                    # Rebuild only the current command slots; do not shift history twice.
                    command_obs = env.commands[:, :3]*env.commands_scale
                    env.obs_buf[:, 6:9] = command_obs
                    env.obs_history[:, -env.num_obs+6:-env.num_obs+9] = command_obs
                    env.eval_force.zero_()
                    if push_end is not None and 3.0 <= time_s < push_end:
                        axis = 1 if scenario == "push_side" else 0
                        sign = -1 if scenario == "push_back" else 1
                        env.eval_force[:, axis] = sign*args.push_force
                    with torch.no_grad():
                        actions, estimates = model.act_inference(env.obs_buf.to(args.rl_device), env.obs_history.to(args.rl_device))
                        estimate_np = {key: value.cpu().numpy() for key, value in estimates.items()}
                        estimate_np["contact_probability"] = estimates["contact_logits"].sigmoid().cpu().numpy()
                        targets = env.estimator_targets.cpu().numpy()
                        metrics.update_estimates(estimate_np, targets)
                        env.step(actions.to(env.device))
                    sample = {key: value.cpu().numpy() for key, value in env.transition_metrics.items()}
                    metrics.update(time_s+env.dt, sample)
                trial_rows = metrics.rows(seed, scenario)
                # Estimator errors were accumulated in normalized units.
                for row in trial_rows:
                    row["velocity_est_rmse_m_s"] *= cfg.estimator.velocity_scale
                    row["slip_est_rmse_m_s"] *= cfg.estimator.slip_scale
                rows.extend(trial_rows)
                scenario_traces[scenario].append((seed, metrics.trace))
                write_csv(output / "{}_seed{}_trace.csv".format(scenario, seed), metrics.trace)
                write_csv(output / "trials.csv", rows)
                print("{} seed={} survival={:.1%}".format(scenario, seed, 1-metrics.failed.mean()))
            finally:
                if env.viewer:
                    env.gym.destroy_viewer(env.viewer)
                env.gym.destroy_sim(env.sim)
        plot_trace(output / (scenario+".png"), scenario_traces[scenario], train_cfg.ablation+" / "+scenario)
    summary = dict(checkpoint=str(checkpoint_path), ablation=saved["train"]["ablation"],
        training_seed=saved["train"]["seed"], training_steps=checkpoint["total_steps"],
        training_protocol=dict(env=saved["env"], policy=saved["train"]["policy"],
                               algorithm=saved["train"]["algorithm"],
                               rollout_steps=saved["train"]["runner"]["num_steps_per_env"]),
        evaluation=dict(protocol_version=3, command_height_m=evaluation_height(cfg),
            reward_version=cfg.rewards.reward_version,
            parking_brake_s=cfg.rewards.parking_brake_s, parking_ramp_s=cfg.rewards.parking_ramp_s,
            parking_deadband_m=cfg.rewards.parking_deadband_m, parking_scale_m=cfg.rewards.parking_scale_m,
            seeds=args.eval_seeds, num_envs=num_envs, duration_s=args.duration,
            pulse_force_n=args.push_force, pulse_start_s=3.0, pulse_duration_s=0.15,
            nominal_friction=0.8, low_friction=0.2, payload_added_mass_kg=0.75, payload_com_offset_x_m=0.02,
            recovery_speed_band_m_s=0.05, recovery_tilt_band_rad=0.12,
            recovery_hold_s=0.5, recovery_deadline_s=3.0, command_speed_band_m_s=0.05,
            command_hold_s=0.5, high_slip_threshold_m_s=0.2, abnormal_spin_body_speed_m_s=0.1),
        note="Metrics cover first episodes until failure; inspect failed and observed_s alongside errors. "
             "std describes evaluation trials, not independent training seeds. "
             "Displacement measures parking drift only in zero-command scenarios.",
        scenarios=summarize(rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    summary_rows = [dict(scenario=scenario, metric=name, **stats)
        for scenario, result in summary["scenarios"].items() for name, stats in result["metrics"].items()]
    write_csv(output / "summary.csv", summary_rows)
    for scenario, result in summary["scenarios"].items():
        m = result["metrics"]
        print("{}: survival={:.1%}, vx RMSE={:.4f} m/s, pitch RMS={:.4f} rad".format(
            scenario, 1-m["failed"]["mean"], m["speed_rmse_m_s"]["mean"], m["pitch_rms_rad"]["mean"]))
    print("Results: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
