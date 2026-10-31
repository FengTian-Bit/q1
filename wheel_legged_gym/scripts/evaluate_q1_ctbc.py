"""Fixed-geometry Q1 CTBC evaluation; helpers also run without Isaac Gym.

``--heights`` is a comma-separated list in centimetres; ``--tread`` is metres.
Official success is consumed from transition infos before the environment resets.
"""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def parse_heights(value):
    """Convert explicit centimetre CLI values to metres, rejecting bad geometry."""
    try:
        heights = [float(item.strip()) / 100.0 for item in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("--heights must be comma-separated centimetres, e.g. 2.5,5,8") from exc
    if not heights or any(not math.isfinite(h) or h <= 0 for h in heights):
        raise ValueError("--heights must contain finite, positive centimetre values")
    return heights


def _list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def guidance_metadata(env):
    """Describe the realized guide and actuator envelope, when available.

    These values describe the configured motion even during pure-policy trials;
    the separate guidance mode and actual feedforward weight determine whether
    that motion contributed to the executed action.
    """
    result = {}
    cfg = getattr(env, "cfg", None)
    ctbc = getattr(cfg, "ctbc", None)
    control = getattr(cfg, "control", None)
    heights = getattr(env, "ctbc_guide_lift_heights", None)
    if heights is not None:
        result["guide_lift_height_m"] = float(heights[0])
    amplitudes = getattr(getattr(env, "guidance", None), "amplitudes", None)
    if amplitudes is not None:
        row = _list(amplitudes[0])
        result["guide_hip_amplitude_rad"] = float(row[0])
        result["guide_knee_amplitude_rad"] = float(row[1])
    duration = getattr(ctbc, "swing_duration", None)
    if duration is not None:
        result["guide_swing_duration_s"] = float(duration)
    start = getattr(ctbc, "feedforward_start_iteration", None)
    if start is not None:
        result["feedforward_start_iteration"] = int(start)
    for key, actual, configured in (
            ("leg_action_scales", "ctbc_leg_scales", "leg_action_scales"),
            ("leg_action_limits", "ctbc_leg_limits", "leg_action_limits"),
            ("action_clip_per_channel", "ctbc_action_clip", None)):
        value = getattr(env, actual, None)
        if value is None and configured is not None:
            value = getattr(control, configured, None)
        if value is not None:
            result[key] = _list(value)
    return result


class EpisodeAccumulator:
    """Count exactly the requested number of completed episodes, without reset races."""

    def __init__(self, requested):
        if requested <= 0:
            raise ValueError("--episodes must be positive")
        self.requested = int(requested)
        self.counts = dict(success=0, physical_failure=0, timeout=0, other=0)

    @property
    def completed(self):
        return sum(self.counts.values())

    def consume(self, infos, dones):
        ended = _list(dones)
        episode = infos.get("ctbc_episode")
        if episode is None:
            if any(ended):
                raise RuntimeError("Completed episode is missing infos['ctbc_episode']; success cannot be inferred after reset")
            return
        ids = _list(episode["env_ids"])
        success = _list(episode["success"])
        failure = _list(episode["physical_failure"])
        if not (len(ids) == len(success) == len(failure)):
            raise ValueError("ctbc_episode arrays must have one entry per completed environment")
        timeout = _list(episode["timeout"]) if "timeout" in episode else None
        all_timeouts = _list(infos["time_outs"]) if "time_outs" in infos else [False] * len(ended)
        if timeout is not None and len(timeout) != len(ids):
            raise ValueError("ctbc_episode timeout length does not match env_ids")
        if len(set(ids)) != len(ids):
            raise ValueError("ctbc_episode contains duplicate environment IDs")
        if set(ids) != {index for index, done in enumerate(ended) if done}:
            raise ValueError("ctbc_episode must include every environment ending on this transition")
        # Stable ordering makes truncation of the final vector step reproducible.
        for index in sorted(range(len(ids)), key=lambda i: ids[i]):
            if self.completed >= self.requested:
                break
            env_id = ids[index]
            if env_id < 0 or env_id >= len(ended) or not ended[env_id]:
                raise ValueError("ctbc_episode must identify environments ending on this transition")
            if success[index] and failure[index]:
                raise ValueError("A successful episode cannot also be a physical failure")
            timed_out = timeout[index] if timeout is not None else all_timeouts[env_id]
            outcome = ("success" if success[index] else "physical_failure" if failure[index]
                       else "timeout" if timed_out else "other")
            self.counts[outcome] += 1

    def summary(self):
        return dict(self.counts, requested=self.requested, completed=self.completed,
                    complete=self.completed == self.requested,
                    success_rate=self.counts["success"] / self.completed if self.completed else None)


def collect_trials(env, policy, episodes, speed, max_steps, inference_context=None):
    """Bounded policy rollout. Deliberately never reads env.ctbc_success after step."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    accumulator = EpisodeAccumulator(episodes)
    steps_run = 0
    with inference_context if inference_context is not None else nullcontext():
        for _ in range(max_steps):
            env.set_command_target(speed, 0.0)
            action = policy(env.obs_buf, env.obs_history)
            if isinstance(action, tuple):
                action = action[0]
            transition = env.step(action)
            accumulator.consume(transition[4], transition[3])
            steps_run += 1
            if accumulator.completed >= episodes:
                break
    return dict(accumulator.summary(), policy_steps=steps_run,
                simulated_seconds=steps_run * env.dt, step_limit=max_steps)


def configure_evaluation(cfg, height, tread, steps, num_envs):
    cfg.env.num_envs = num_envs
    cfg.terrain.curriculum = False
    cfg.terrain.selected = True
    cfg.terrain.selected_height = height
    cfg.terrain.selected_tread = tread
    cfg.terrain.selected_steps = steps
    cfg.terrain.selected_flat = False
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    cfg.init_state.reset_lin_vel = cfg.init_state.reset_ang_vel = 0.0


def evaluation_arguments():
    return [
        {"name": "--heights", "type": str, "default": "2.5,5,8,10,12",
         "help": "Comma-separated riser heights in CENTIMETRES, e.g. 2.5,5,8."},
        {"name": "--tread", "type": float, "default": 0.5,
         "help": "Stair tread width in metres."},
        {"name": "--steps", "type": int, "default": 1,
         "help": "Number of ascending risers; use 1 for single-step, 3 for stairs."},
        {"name": "--episodes", "type": int, "default": 100,
         "help": "Exact number of completed episodes per height and guidance condition."},
        {"name": "--speed", "type": float, "default": 0.2,
         "help": "Forward command in metres per second; yaw command is zero."},
        {"name": "--guidance", "type": str, "choices": ["off", "on", "both"], "default": "off",
         "help": "off evaluates pure policy; on restores training feedforward as a diagnostic."},
        {"name": "--output", "type": str, "default": None, "help": "Output JSON path."},
        {"name": "--eval_max_steps", "type": int, "default": 0,
         "help": "Maximum policy steps per condition; 0 derives a bound from episode length."},
    ]


def evaluate(args, export_policy=False, plot_torques=False):
    # Isaac Gym must precede torch imports. Importing pure helpers needs neither.
    import isaacgym  # noqa: F401
    import torch
    from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
    import wheel_legged_gym.envs  # noqa: F401 -- registers tasks
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.utils.helpers import class_to_dict, get_load_path
    from wheel_legged_gym.envs.q1_ctbc.run_config import apply_config, validate_ctbc_snapshot
    from wheel_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner

    if args.task != "q1_ctbc":
        raise ValueError("This evaluator requires --task=q1_ctbc")
    export_only = getattr(args, "export_only", False)
    export_policy = export_policy or export_only or getattr(args, "export_onnx", False)
    heights = parse_heights(getattr(args, "heights", "5"))
    tread, steps = getattr(args, "tread", 0.5), getattr(args, "steps", 1)
    episodes, speed = getattr(args, "episodes", 100), getattr(args, "speed", 0.2)
    if not math.isfinite(tread) or tread <= 0 or steps <= 0 or episodes <= 0:
        raise ValueError("tread, steps and episodes must be positive")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("--speed must be finite and positive for ascending stairs")
    if getattr(args, "eval_max_steps", 0) < 0:
        raise ValueError("--eval_max_steps must be nonnegative")
    guidance = getattr(args, "guidance", "off")
    if guidance not in ("off", "on", "both"):
        raise ValueError("--guidance must be off, on or both")
    num_envs = 64 if args.num_envs is None else args.num_envs
    if num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    base_cfg, base_train = (deepcopy(item) for item in task_registry.get_cfgs("q1_ctbc"))
    experiment = args.experiment_name or base_train.runner.experiment_name
    checkpoint = Path(get_load_path(str(Path(WHEEL_LEGGED_GYM_ROOT_DIR) / "logs" / experiment),
                                   args.load_run if args.load_run is not None else -1,
                                   args.checkpoint if args.checkpoint is not None else -1)).resolve()
    snapshot_path = checkpoint.parent / "resolved_config.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        validate_ctbc_snapshot(snapshot)
        apply_config(base_cfg, snapshot["env"])
        apply_config(base_train, snapshot["train"])
    else:
        raise FileNotFoundError("Evaluation requires the checkpoint's resolved_config.json: " + str(snapshot_path))
    seed = args.seed if args.seed is not None else base_train.seed
    conditions = [False, True] if guidance == "both" else [guidance == "on"]
    report = dict(task="q1_ctbc", checkpoint=str(checkpoint), seed=seed,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  geometry_units="metres", speed_mps=speed, yaw_rate=0.0,
                  num_envs=num_envs, requested_episodes_per_condition=episodes,
                  config_snapshot=str(snapshot_path), export_only=export_only, results=[])
    output = getattr(args, "output", None)
    if output is None:
        output = checkpoint.parent / "evaluations" / ("ctbc_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for height in heights:
            for guided in conditions:
                cfg, train_cfg, trial_args = deepcopy(base_cfg), deepcopy(base_train), deepcopy(args)
                configure_evaluation(cfg, height, tread, steps, num_envs)
                cfg.seed = trial_args.seed = seed
                trial_args.num_envs = num_envs
                env, torque_recorder = None, None
                try:
                    env, _ = task_registry.make_env("q1_ctbc", args=trial_args, env_cfg=cfg)
                    runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
                    runner.load(str(checkpoint), load_optimizer=False)
                    # Export once per checkpoint, before any potentially long
                    # rollout. Later interrupts cannot prevent this artifact.
                    if export_policy and "policy_onnx" not in report:
                        from wheel_legged_gym.rsl_rl.modules.policy_exporter import export_policy_as_onnx
                        policy_path = checkpoint.parent / "exported" / checkpoint.stem / "policy.onnx"
                        policy_path = export_policy_as_onnx(
                            runner.alg.actor_critic, policy_path,
                            num_observations=cfg.env.num_observations,
                            num_history=cfg.env.num_observations * cfg.env.obs_history_length,
                        )
                        report["policy_onnx"] = str(policy_path)
                        print("Exported Q1 CTBC policy as ONNX:", policy_path, flush=True)
                    if export_only:
                        return report
                    if plot_torques:
                        from wheel_legged_gym.utils.torque_plot import TorqueRecorder
                        torque_recorder = TorqueRecorder(env)
                        env._play_torque_recorder = torque_recorder
                    # Force a diagnostic mode after load, which restores cumulative iteration.
                    env.set_guidance_mode("on" if guided else "off")
                    policy = runner.get_inference_policy(device=env.device)
                    limit = getattr(args, "eval_max_steps", 0) or (
                        (math.ceil(episodes / num_envs) + 1) * (int(env.max_episode_length) + 1))
                    result = collect_trials(env, policy, episodes, speed, limit, torch.inference_mode())
                    result.update(requested_height_m=height, requested_tread_m=tread,
                                  height_m=float(env.terrain.riser_heights[0, 0]),
                                  tread_m=float(env.terrain.tread_widths[0, 0]),
                                  stair_steps=int(env.terrain.step_counts[0, 0]),
                                  first_riser_x_m=float(env.terrain.first_riser_x[0, 0]),
                                  goal_x_m=float(env.terrain.goal_x[0, 0]),
                                  guidance="on" if guided else "off", pure_policy=not guided,
                                  feedforward_weight=float(env.ctbc_feedforward_weight),
                                  checkpoint_iteration=int(runner.current_learning_iteration),
                                  physics_dt=float(cfg.sim.dt), policy_dt=float(env.dt),
                                  horizontal_scale=float(cfg.terrain.horizontal_scale),
                                  vertical_scale=float(cfg.terrain.vertical_scale),
                                  observation_noise=bool(cfg.noise.add_noise),
                                  domain_randomization=class_to_dict(cfg.domain_rand))
                    result.update(guidance_metadata(env))
                    report["results"].append(result)
                    print("height={:.1f} cm guidance={} success={}/{} ({})".format(
                        height * 100, result["guidance"], result["success"], result["completed"],
                        "complete" if result["complete"] else "STEP LIMIT REACHED"), flush=True)
                finally:
                    try:
                        if torque_recorder is not None:
                            env._play_torque_recorder = None
                            plot_path = output.with_name(output.stem + "_h{:g}cm_{}_torques.png".format(
                                height * 100, "guided" if guided else "policy"))
                            saved = torque_recorder.save_png(plot_path, title=(
                                "Q1 CTBC | {} | {:g} cm | Guidance {}".format(
                                    checkpoint.stem, height * 100, "on" if guided else "off")))
                            if saved is not None:
                                report.setdefault("torque_plots", []).append(str(saved))
                    finally:
                        if env is not None:
                            if env.viewer is not None:
                                env.gym.destroy_viewer(env.viewer)
                            env.gym.destroy_sim(env.sim)
                # Preserve completed conditions even if a later simulation cannot start.
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        report["stop_reason"] = "keyboard_interrupt"
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        report["stop_reason"] = "viewer_closed"
    finally:
        report["complete"] = ("policy_onnx" in report if export_only else (
            len(report["results"]) == len(heights) * len(conditions)
            and all(row["complete"] for row in report["results"])))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Q1 CTBC export JSON:" if export_only else "Q1 CTBC evaluation JSON:", output, flush=True)
    return report


def main():
    import isaacgym  # noqa: F401
    # Register environments before utils imports TaskRegistry, which itself
    # imports envs.base. Match the working train/play entry-point order.
    import wheel_legged_gym.envs  # noqa: F401
    from wheel_legged_gym.utils import get_args
    args = get_args(extra_custom_parameters=evaluation_arguments(), default_task="q1_ctbc")
    report = evaluate(args)
    if not report["complete"]:
        raise SystemExit("Evaluation did not reach the requested number of completed episodes")


if __name__ == "__main__":
    main()
