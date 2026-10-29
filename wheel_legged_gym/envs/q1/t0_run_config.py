"""Restore Q1's T0-terrain training contract before creating the simulator."""
from copy import deepcopy
import json
from pathlib import Path
import re


def apply_config(target, values):
    """Restore saved values while keeping nested configuration objects intact."""
    for key, value in values.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            apply_config(current, value)
        else:
            setattr(target, key, value)


def _validate_ctbc_snapshot(snapshot):
    # Lazy imports keep this module usable by CPU-only configuration tools.
    from wheel_legged_gym.envs.q1_ctbc.run_config import validate_ctbc_snapshot
    return validate_ctbc_snapshot(snapshot)


def validate_q1_t0_snapshot(snapshot):
    """Reject legacy Q1 policies whose identically shaped actions mean something else."""
    error = "Cannot resume q1: incompatible T0-terrain resolved_config.json; "
    if not isinstance(snapshot, dict):
        raise ValueError(error + "the snapshot must be an object")
    env = snapshot.get("env")
    if (snapshot.get("task") != "q1" or snapshot.get("stage") != "t0"
            or not isinstance(env, dict)
            or not isinstance(env.get("ctbc"), dict)
            or env["ctbc"].get("stage") != "t0"
            or not isinstance(env.get("terrain"), dict)
            or env["terrain"].get("generation_mode") != "t0"):
        raise ValueError(error + "expected task=q1, stage=t0, env.ctbc.stage=t0 and "
                         "env.terrain.generation_mode=t0; old q1 checkpoints are not resumable")
    # Reuse the CTBC observation/action checks without accepting its corridor
    # geometry as a new Q1 experiment. Only the validation copy changes stage.
    contract = deepcopy(snapshot)
    contract["task"] = "q1_ctbc"
    contract["stage"] = contract["env"]["ctbc"]["stage"] = "stairs"
    try:
        _validate_ctbc_snapshot(contract)
    except ValueError as exc:
        raise ValueError(error + str(exc)) from exc
    return "t0"


def _configure_q1_t0(cfg):
    from .q1_t0_config import configure_q1_t0
    return configure_q1_t0(cfg)


def _load_checkpoint_iteration(checkpoint):
    """The saved PPO iteration, rather than the filename, starts migration guidance."""
    import torch

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    iteration = state.get("iter") if isinstance(state, dict) else None
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Cannot migrate to q1: checkpoint 'iter' must be a nonnegative integer")
    return iteration


def prepare_q1_t0_training(env_cfg, train_cfg, args, root_dir, get_load_path):
    """Resume the same task, or explicitly migrate an absolute CTBC run path.

    ``experiment_name`` keeps its existing meaning as an output override. CTBC
    sources use an absolute ``load_run`` so their experiment name cannot redirect
    new Q1 outputs into the old task's log directory. Standard registry handling
    still applies CLI seed, environment count, update count and logger overrides.
    """
    if getattr(args, "q1_stage", None) is not None:
        raise ValueError("--task=q1 now uses the fixed T0 terrain curriculum; "
                         "--q1_stage is supported only by q1_flat")
    if getattr(args, "ctbc_stage", None) is not None:
        raise ValueError("--ctbc_stage requires --task=q1_ctbc; --task=q1 uses the T0 terrain curriculum")
    output_experiment = getattr(args, "experiment_name", None)
    if output_experiment == "q1_ctbc":
        raise ValueError("For CTBC migration, remove --experiment_name=q1_ctbc and supply "
                         "the full absolute --load_run path; new q1 logs default to logs/q1")
    migrated = False
    if getattr(args, "resume", False):
        run = getattr(args, "load_run", None)
        run = -1 if run is None or run == "-1" else run
        absolute_run = isinstance(run, (str, Path)) and Path(run).is_absolute()
        experiment = output_experiment or "q1"
        # get_load_path enumerates its root even for absolute run paths. Use the
        # source's existing parent when logs/q1 has not been created yet.
        source_root = Path(run).parent if absolute_run else Path(root_dir) / "logs" / experiment
        number = getattr(args, "checkpoint", None)
        checkpoint = Path(get_load_path(str(source_root), load_run=run,
                                        checkpoint=-1 if number is None else number)).resolve()
        match = re.fullmatch(r"model_(0|[1-9]\d*)\.pt", checkpoint.name)
        if match is None:
            raise ValueError("Cannot resume q1: expected a model_N.pt checkpoint, found " + str(checkpoint))
        if not checkpoint.is_file():
            raise FileNotFoundError("Cannot resume q1: checkpoint does not exist: " + str(checkpoint))
        config_path = checkpoint.parent / "resolved_config.json"
        if not config_path.is_file():
            raise FileNotFoundError("Cannot resume q1 without its saved configuration: " + str(config_path))
        try:
            snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read q1 configuration: " + str(config_path)) from exc
        if not isinstance(snapshot, dict):
            raise ValueError("Cannot resume q1: the saved configuration must be an object")
        if snapshot.get("task") == "q1":
            validate_q1_t0_snapshot(snapshot)
        elif snapshot.get("task", "q1_ctbc") == "q1_ctbc":
            _validate_ctbc_snapshot(snapshot)
            if not absolute_run:
                raise ValueError("Migrating q1_ctbc to q1 requires the full absolute --load_run path")
            migrated = True
        else:
            raise ValueError("Cannot resume q1 from task {!r}".format(snapshot.get("task")))
        # Read before mutating configs so corrupt migration metadata fails early.
        entry_iteration = _load_checkpoint_iteration(checkpoint) if migrated else None
        apply_config(env_cfg, snapshot["env"])
        apply_config(train_cfg, snapshot["train"])
        if "feedforward_start_iteration" not in snapshot["env"]["ctbc"]:
            env_cfg.ctbc.feedforward_start_iteration = 0
        if migrated:
            _configure_q1_t0(env_cfg)
            env_cfg.ctbc.feedforward_start_iteration = entry_iteration
            train_cfg.runner.ctbc_guidance_entry_checkpoint_iteration = entry_iteration
            train_cfg.runner.q1_t0_source_task = "q1_ctbc"
            train_cfg.runner.q1_t0_source_checkpoint = str(checkpoint)
        args.load_run = str(checkpoint.parent)
        args.checkpoint = int(match.group(1))
        train_cfg.runner.load_run = args.load_run
        train_cfg.runner.checkpoint = args.checkpoint
        train_cfg.runner.resume = True
    else:
        _configure_q1_t0(env_cfg)
        env_cfg.ctbc.feedforward_start_iteration = 0
        train_cfg.runner.resume = False
    train_cfg.runner.experiment_name = output_experiment or "q1"
    train_cfg.runner.ctbc_guidance_start_iteration = getattr(env_cfg.ctbc, "feedforward_start_iteration", 0)
    default_name = "t0" if migrated or not getattr(args, "resume", False) else train_cfg.runner.run_name
    args.run_name = getattr(args, "run_name", None) or default_name
    train_cfg.runner.run_name = args.run_name
    return env_cfg, train_cfg
