"""Checkpoint-faithful, pure-policy playback for Q1's T0-terrain task."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace


def _runtime():
    """Load simulator dependencies only when playback is actually requested."""
    from isaacgym import gymapi
    import torch

    from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.utils.helpers import class_to_dict, get_load_path
    from wheel_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from wheel_legged_gym.envs.q1.t0_run_config import prepare_q1_t0_training
    from wheel_legged_gym.envs.q1.keyboard import Q1Keyboard

    return SimpleNamespace(
        torch=torch, gymapi=gymapi, root=WHEEL_LEGGED_GYM_ROOT_DIR,
        registry=task_registry, class_to_dict=class_to_dict,
        get_load_path=get_load_path, runner=OnPolicyRunner,
        prepare=prepare_q1_t0_training, keyboard=Q1Keyboard,
    )


def _values(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return list(value)


def _collect_episode(report, infos, dones):
    """Count completed T0 episodes; distance progress is not climbing success."""
    event = infos.get("q1_t0_episode")
    if not event:
        return
    ids = _values(event.get("env_ids", []))
    failed = _values(event.get("physical_failure", []))
    timeout = _values(event.get("timeout", []))
    distances = _values(event.get("distance_m", []))
    if not (len(ids) == len(failed) == len(timeout) == len(distances)):
        raise ValueError("Q1 T0 playback received misaligned episode statistics")
    done_values = _values(dones)
    if len(set(ids)) != len(ids):
        raise ValueError("Q1 T0 playback received duplicate completed environment IDs")
    for index, failure, timed_out, distance in zip(ids, failed, timeout, distances):
        if not 0 <= index < len(done_values):
            raise ValueError("Q1 T0 playback received an invalid completed environment ID")
        if not done_values[index]:
            continue
        report["completed_episodes"] += 1
        report["physical_failures"] += int(bool(failure))
        report["timeouts"] += int(bool(timed_out) and not bool(failure))
        report["other_terminations"] += int(not failure and not timed_out)
        report["completed_distance_sum_m"] += float(distance)


def _refresh_keyboard_observation(env):
    """Expose this frame's manual command without advancing history twice."""
    env.obs_buf[:] = env.compute_proprioception_observations().clamp(
        -env.cfg.normalization.clip_observations,
        env.cfg.normalization.clip_observations,
    )
    env.obs_history[:, -env.num_obs:] = env.obs_buf


def play_q1(args, playback_configure=None):
    """Restore a new-Q1 checkpoint and play its deterministic policy means.

    Automatic playback retains T0's random command distribution and heading
    controller. Keyboard playback supplies direct vx/yaw targets. Both remove
    training guidance and report completed distance episodes, not stair success.
    """
    if getattr(args, "task", "q1") != "q1":
        raise ValueError("This playback entry requires --task=q1")
    duration = float(getattr(args, "play_duration", 0.0))
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("--play_duration must be finite and nonnegative")
    mode = getattr(args, "play_mode", "keyboard")
    if getattr(args, "headless", False):
        mode = "auto"
    if mode not in ("auto", "keyboard"):
        raise ValueError("--play_mode must be keyboard or auto")
    if mode == "auto" and duration == 0:
        duration = 60.0

    runtime = _runtime()
    play_args = deepcopy(args)
    play_args.resume = True
    cfg, train_cfg = (deepcopy(value) for value in runtime.registry.get_cfgs("q1"))
    cfg, train_cfg = runtime.prepare(
        cfg, train_cfg, play_args, runtime.root, runtime.get_load_path
    )
    # The prepare step rejects legacy Q1 contracts and pins an exact checkpoint.
    checkpoint = Path(play_args.load_run) / "model_{}.pt".format(play_args.checkpoint)
    if playback_configure is not None:
        cfg = playback_configure(cfg, play_args)
    count = 16 if getattr(play_args, "num_envs", None) is None else play_args.num_envs
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("--num_envs must be a positive integer")
    cfg.env.num_envs = play_args.num_envs = count
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if name.startswith("randomize_") or name == "push_robots":
            setattr(cfg.domain_rand, name, False)
    cfg.init_state.reset_lin_vel = cfg.init_state.reset_ang_vel = 0.0
    cfg.ctbc.pure_policy = True

    output = getattr(play_args, "output", None)
    if output is None:
        output = checkpoint.parent / "evaluations" / (
            "q1_t0_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "q1", "stage": cfg.ctbc.stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "config_snapshot": str(checkpoint.parent / "resolved_config.json"),
        "play_mode": mode, "num_envs": count,
        "guidance_mode": "off", "feedforward_weight": 0.0,
        "requested_duration_s": duration,
        "policy_steps": 0, "completed_episodes": 0,
        "physical_failures": 0, "timeouts": 0, "other_terminations": 0,
        "completed_distance_sum_m": 0.0,
        "playback_config": runtime.class_to_dict(cfg),
    }
    env, controls = None, None
    try:
        env, cfg = runtime.registry.make_env("q1", args=play_args, env_cfg=cfg)
        report["playback_config"] = runtime.class_to_dict(cfg)
        env.external_commands = mode == "keyboard"
        # Runner construction resets the simulator. Disable guidance before
        # that reset as well as after loading its cumulative training iteration.
        env.set_guidance_mode("off")
        runner = runtime.runner(
            env, runtime.class_to_dict(train_cfg), log_dir=None,
            device=getattr(play_args, "rl_device", env.device),
        )
        runner.load(str(checkpoint), load_optimizer=False)
        env.set_guidance_mode("off")
        env.reset()
        policy = runner.get_inference_policy(device=env.device)
        report["checkpoint_iteration"] = int(runner.current_learning_iteration)
        report["feedforward_start_iteration"] = int(
            getattr(cfg.ctbc, "feedforward_start_iteration", 0)
        )
        report["feedforward_weight"] = float(env.ctbc_feedforward_weight)
        if mode == "keyboard":
            controls = runtime.keyboard(env)
            controls.enable(runtime.gymapi)
        while duration == 0 or report["policy_steps"] * env.dt < duration - 1e-9:
            with runtime.torch.inference_mode():
                if controls is not None:
                    controls.render()
                    if controls.paused:
                        time.sleep(0.02)
                        continue
                    controls.apply()
                    env.commands[:] = env.command_targets
                    _refresh_keyboard_observation(env)
                result = policy(env.obs_buf, env.obs_history)
                actions = result[0] if isinstance(result, tuple) else result
                _, _, _, dones, infos, _ = env.step(actions)
                _collect_episode(report, infos, dones)
                report["policy_steps"] += 1
    except SystemExit:
        # Closing the viewer still writes the partial report and restores keys.
        report["viewer_closed"] = True
    finally:
        try:
            if controls is not None:
                controls.close()
            report["simulated_duration_s"] = (
                report["policy_steps"] * env.dt if env is not None else 0.0
            )
            completed = report["completed_episodes"]
            report["mean_completed_distance_m"] = (
                report["completed_distance_sum_m"] / completed if completed else None
            )
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Q1 T0 playback report:", output)
        finally:
            if env is not None:
                if env.viewer is not None:
                    env.gym.destroy_viewer(env.viewer)
                env.gym.destroy_sim(env.sim)
    return report
