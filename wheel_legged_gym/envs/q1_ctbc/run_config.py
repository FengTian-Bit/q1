"""Restore the saved CTBC experiment before creating environments or optimizers."""
import json
import math
from pathlib import Path
import re


STAGES = ("single", "stairs", "robust", "stairs20")


def apply_config(target, values):
    """Merge a JSON configuration into the already-instantiated config classes."""
    for key, value in values.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            apply_config(current, value)
        else:
            setattr(target, key, value)


def validate_ctbc_snapshot(snapshot, task_name="q1_ctbc"):
    """Reject other tasks and incomplete observation/action contracts explicitly."""
    error = "Cannot resume {}: incompatible resolved_config.json; ".format(task_name)
    if not isinstance(snapshot, dict):
        raise ValueError(error + "the snapshot must be an object")
    env, train = snapshot.get("env"), snapshot.get("train")
    if not isinstance(env, dict) or not isinstance(env.get("ctbc"), dict) or not isinstance(train, dict):
        raise ValueError(error + "a q1_ctbc snapshot with env.ctbc and train is required; q1 checkpoints are not resumable")
    if snapshot.get("task", "q1_ctbc") != task_name:
        raise ValueError(error + "the saved task is not " + task_name)
    if task_name == "q1_ctbc_highcom" and env["ctbc"].get("stance_profile") != "highcom_knee45_v1":
        raise ValueError(error + "the saved high-COM stance profile is missing or incompatible")
    expected = {"num_observations": 31, "num_privileged_obs": 151,
                "obs_history_length": 5, "num_actions": 6}
    dimensions = env.get("env", {})
    if not isinstance(dimensions, dict):
        raise ValueError(error + "env.env must describe the observation/action interface")
    for name, required in expected.items():
        if dimensions.get(name) != required:
            raise ValueError(error + "env.env.{} must be {}, found {!r}".format(name, required, dimensions.get(name)))
    policy = train.get("policy", {})
    if not isinstance(policy, dict) or policy.get("num_encoder_obs") != 155:
        raise ValueError(error + "train.policy.num_encoder_obs must be 155 (31 observations × 5 frames)")
    control = env.get("control", {})
    scales = control.get("leg_action_scales") if isinstance(control, dict) else None
    if (not isinstance(scales, (list, tuple)) or len(scales) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in scales)):
        raise ValueError(error + "env.control.leg_action_scales must contain four finite positive values")
    stage = env["ctbc"].get("stage")
    if stage not in STAGES:
        raise ValueError(error + "env.ctbc.stage must be single, stairs, robust or stairs20")
    start_iteration = env["ctbc"].get("feedforward_start_iteration", 0)
    if isinstance(start_iteration, bool) or not isinstance(start_iteration, int) or start_iteration < 0:
        raise ValueError(error + "env.ctbc.feedforward_start_iteration must be a nonnegative integer")
    if snapshot.get("stage", stage) != stage:
        raise ValueError(error + "top-level stage and env.ctbc.stage disagree")
    if not isinstance(train.get("runner"), dict):
        raise ValueError(error + "train.runner is missing")
    return stage


def _configure_stage(cfg, stage):
    # Keep snapshot validation and tests independent of Isaac Gym imports.
    from .q1_ctbc_config import configure_ctbc_stage
    return configure_ctbc_stage(cfg, stage)


def _load_checkpoint_iteration(checkpoint):
    """Read the saved update count when entering a new high-stair curriculum.

    The filename is not authoritative: older checkpoints can have a saved
    iteration that differs from their name. Restrict deserialization to tensors
    and plain metadata rather than enabling arbitrary checkpoint objects.
    """
    import torch

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    iteration = state.get("iter") if isinstance(state, dict) else None
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Cannot enter stairs20: checkpoint 'iter' must be a nonnegative integer")
    return iteration


def prepare_ctbc_training(env_cfg, train_cfg, args, root_dir, get_load_path,
                          *, task_name="q1_ctbc", stage_configurator=None):
    """Restore a checkpoint config, then apply any explicitly requested stage.

    Other CLI overrides remain untouched for the registry's standard override
    mechanism. Resolving the checkpoint now prevents a later "latest" lookup
    from silently selecting a different run or model.
    """
    configure_stage = stage_configurator or _configure_stage
    requested_stage = getattr(args, "ctbc_stage", None)
    if requested_stage is not None and requested_stage not in STAGES:
        raise ValueError("Unknown CTBC stage: {}".format(requested_stage))
    if getattr(args, "resume", False):
        experiment = getattr(args, "experiment_name", None) or train_cfg.runner.experiment_name
        run = getattr(args, "load_run", None)
        checkpoint_number = getattr(args, "checkpoint", None)
        run = -1 if run is None or run == "-1" else run
        checkpoint_number = -1 if checkpoint_number is None else checkpoint_number
        checkpoint = Path(get_load_path(str(Path(root_dir) / "logs" / experiment),
                                        load_run=run, checkpoint=checkpoint_number)).resolve()
        match = re.fullmatch(r"model_(0|[1-9]\d*)\.pt", checkpoint.name)
        if match is None:
            raise ValueError("Cannot resume {}: expected a model_N.pt checkpoint, found ".format(task_name) + str(checkpoint))
        if not checkpoint.is_file():
            raise FileNotFoundError("Cannot resume {}: checkpoint does not exist: ".format(task_name) + str(checkpoint))
        config_path = checkpoint.parent / "resolved_config.json"
        if not config_path.is_file():
            raise FileNotFoundError("Cannot resume {} without its saved configuration: ".format(task_name) + str(config_path))
        try:
            snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read {} configuration: ".format(task_name) + str(config_path)) from exc
        saved_stage = validate_ctbc_snapshot(snapshot, task_name=task_name)
        apply_config(env_cfg, snapshot["env"])
        apply_config(train_cfg, snapshot["train"])
        if "feedforward_start_iteration" not in snapshot["env"]["ctbc"]:
            # Legacy snapshots predate a stage-relative guidance clock.
            env_cfg.ctbc.feedforward_start_iteration = 0
        stage = requested_stage or saved_stage
        saved_start = getattr(env_cfg.ctbc, "feedforward_start_iteration", 0)
        # An explicit stage reapplies current curriculum presets even when its
        # name is unchanged. Omitting it preserves all saved tuning parameters.
        if requested_stage is not None:
            configure_stage(env_cfg, stage)
        if stage == "stairs20":
            if saved_stage != "stairs20":
                # Start a fresh guidance schedule at the loaded policy's real
                # cumulative update, without changing its model/action contract.
                entry_iteration = _load_checkpoint_iteration(checkpoint)
                env_cfg.ctbc.feedforward_start_iteration = entry_iteration
                train_cfg.runner.ctbc_guidance_entry_checkpoint_iteration = entry_iteration
            else:
                # Explicit same-stage requests may refresh terrain presets, but
                # must not restart guidance after every training interruption.
                env_cfg.ctbc.feedforward_start_iteration = saved_start
        elif saved_stage == "stairs20":
            # Returning to an earlier curriculum restores its original global
            # cumulative-iteration schedule rather than inheriting a high-stage
            # relative clock. Switching between old stages still never restarts.
            env_cfg.ctbc.feedforward_start_iteration = 0
        args.load_run = str(checkpoint.parent)
        args.checkpoint = int(match.group(1))
        train_cfg.runner.load_run = args.load_run
        train_cfg.runner.checkpoint = args.checkpoint
        train_cfg.runner.resume = True
    else:
        stage = requested_stage or "single"
        configure_stage(env_cfg, stage)
        if stage == "stairs20":
            env_cfg.ctbc.feedforward_start_iteration = 0
    args.ctbc_stage = stage
    train_cfg.runner.ctbc_guidance_start_iteration = getattr(env_cfg.ctbc, "feedforward_start_iteration", 0)
    args.run_name = getattr(args, "run_name", None) or stage
    train_cfg.runner.run_name = args.run_name
    return env_cfg, train_cfg
