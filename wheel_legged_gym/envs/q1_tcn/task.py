"""Public q1-style train/play entry points for the fixed standalone TCN task."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import platform
import shutil


def runtime():
    import torch
    from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.utils.helpers import get_load_path
    from .task_config import Q1TCNTaskCfg, Q1TCNTaskCfgPPO, PROFILE_VERSION, validate_task_config
    from .runner import restore_config, load_checkpoint, config_dict
    from .joint_runner import JointTCNPpoRunner
    return SimpleNamespace(torch=torch, root=Path(WHEEL_LEGGED_GYM_ROOT_DIR),
        registry=task_registry, get_load_path=get_load_path, env_config=Q1TCNTaskCfg,
        train_config=Q1TCNTaskCfgPPO, version=PROFILE_VERSION, validate=validate_task_config,
        restore=restore_config, load=load_checkpoint, plain=config_dict, runner=JointTCNPpoRunner)


def checkpoint_path(args, rt):
    experiment = getattr(args, "experiment_name", None) or "q1_tcn"
    run = getattr(args, "load_run", None)
    run = -1 if run is None or run == "-1" else run
    root = Path(run).parent if isinstance(run, str) and Path(run).is_absolute() else rt.root/"logs"/experiment
    number = getattr(args, "checkpoint", None)
    path = Path(rt.get_load_path(str(root), load_run=run,
                   checkpoint=-1 if number is None else number)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare(args, rt, resume=None, for_play=False):
    cfg, train = rt.env_config(), rt.train_config()
    checkpoint = None
    resume = getattr(args, "resume", False) if resume is None else resume
    if getattr(args, "q1_stage", None) or getattr(args, "ctbc_stage", None):
        raise ValueError("q1_tcn jointly trains parking and motion; no CTBC/q1 stage flag is used")
    if resume:
        checkpoint = checkpoint_path(args, rt)
        saved = rt.load(checkpoint)["config"]
        if saved.get("task") != "q1_tcn" or saved.get("profile_version") != rt.version:
            raise ValueError("Expected a standalone joint checkpoint from " + rt.version)
        rt.restore(cfg, saved["env"])
        rt.restore(train, saved["train"])
    for source, target, field in (
        ("num_envs", cfg.env, "num_envs"), ("seed", train, "seed"),
        ("max_iterations", train.runner, "max_iterations"),
        ("experiment_name", train.runner, "experiment_name"),
        ("run_name", train.runner, "run_name"), ("logger", train.runner, "logger"),
        ("swanlab_project", train.runner, "swanlab_project"),
        ("swanlab_mode", train.runner, "swanlab_mode")):
        value = getattr(args, source, None)
        if value is not None:
            setattr(target, field, value)
    cfg.seed = train.seed
    if train.runner.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    rt.validate(cfg, train, training=not for_play)
    return cfg, train, checkpoint


def snapshot_sources(output, rt):
    """Keep code, robot geometry and hashes together; never copy model/log trees."""
    paths = [rt.root/"setup.py"]
    paths += sorted((rt.root/"wheel_legged_gym").rglob("*.py"))
    paths += sorted(path for path in (rt.root/"resources/robots/q1").rglob("*") if path.is_file())
    manifest = {}
    for source in paths:
        relative = source.relative_to(rt.root)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output/"source_manifest.json").write_text(json.dumps(dict(
        profile_version=rt.version, files=manifest), indent=2), encoding="utf-8")


def create_writer(output, train, rt):
    if train.runner.logger == "swanlab":
        from wheel_legged_gym.rsl_rl.utils.swanlab_utils import SwanLabSummaryWriter
        writer = SwanLabSummaryWriter(str(output), 10, rt.plain(train.runner))
    elif train.runner.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output))
    else:
        raise ValueError("q1_tcn logger must be swanlab or tensorboard")
    return writer


def train_task(args, rt=None):
    rt = rt or runtime()
    cfg, train, checkpoint = prepare(args, rt)
    rt.torch.set_num_threads(1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = rt.root/"logs"/train.runner.experiment_name/(stamp+"_"+train.runner.run_name)
    output.mkdir(parents=True, exist_ok=False)
    env, writer, runner = None, None, None
    try:
        env, _ = rt.registry.make_env("q1_tcn", args=args, env_cfg=cfg)
        runner = rt.runner(env, train, output, device=args.rl_device)
        env.reset()
        if checkpoint is not None:
            runner.load(checkpoint)
        snapshot_sources(output, rt)
        (output/"initialization.json").write_text(json.dumps(dict(
            profile_version=rt.version, initialization="resume" if checkpoint else "random",
            checkpoint=str(checkpoint) if checkpoint else None,
            parking_trainable=True, motion_trainable=True,
            python=platform.python_version(), torch=rt.torch.__version__,
            cuda=rt.torch.version.cuda), indent=2), encoding="utf-8")
        writer = create_writer(output, train, rt)
        if hasattr(writer, "store_config"):
            writer.store_config(cfg, train, dict(checkpoint=str(checkpoint) if checkpoint else None,
                                                profile_version=rt.version))
        runner.writer = writer
        runner.save()
        print("q1_tcn {} | parking + motion | {}".format(rt.version, output), flush=True)
        runner.learn(train.runner.max_iterations)
    except KeyboardInterrupt:
        if runner is not None:
            print("Interrupted; saved {}".format(runner.save()), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            if env.viewer:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    return output


def play_task(args, rt=None):
    # Reuse the established keyboard/play loop with a task-specific config
    # restorer, so the public task never enters the legacy generic PPO runner.
    rt = rt or runtime()
    from . import play as legacy_play
    cfg, train, checkpoint = prepare(args, rt, resume=True, for_play=True)
    from .runner import load_checkpoint
    saved = load_checkpoint(checkpoint)
    mode = getattr(args, "play_mode", None) or "keyboard"
    mode = "combined" if mode == "auto" else mode
    duration = getattr(args, "play_duration", None) or 600.0
    options = SimpleNamespace(**vars(args))
    options.checkpoint_path, options.mode = str(checkpoint), mode
    options.duration = duration
    options.tracking_speed, options.yaw_speed = 1.2, 1.0

    def build_scene(snapshot, scenario, seed, num_envs, horizon):
        env_cfg, train_cfg = rt.env_config(), rt.train_config()
        rt.restore(env_cfg, snapshot["env"])
        rt.restore(train_cfg, snapshot["train"])
        env_cfg.seed, env_cfg.env.num_envs = seed, num_envs
        env_cfg.env.episode_length_s = horizon + 2.0
        env_cfg.commands.resampling_time = horizon + 10.0
        env_cfg.commands.parking_env_fraction = 0.0
        env_cfg.commands.sequence_fraction = 0.0
        return env_cfg, train_cfg

    # Explicit arguments preserve the existing module CLI for old experiments.
    return legacy_play.run(options, saved=saved, scene_builder=build_scene)
