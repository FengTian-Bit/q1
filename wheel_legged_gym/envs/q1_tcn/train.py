"""Run from the repository root with python -m ...q1_tcn.train."""

from datetime import datetime
from pathlib import Path
from shutil import copy2

import isaacgym  # Must precede torch imports in Isaac Gym installations.

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils import get_args, task_registry
from .q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO, validate_config
from .runner import TCNPpoRunner
from .cli import get_tcn_args


def main():
    args = get_tcn_args(get_args)
    if args.task != "q1_tcn":
        raise ValueError("This entry point only supports --task=q1_tcn")
    if args.resume:
        raise ValueError("Use --resume_checkpoint PATH for q1_tcn")
    env_cfg, train_cfg = Q1TCNCfg(), Q1TCNCfgPPO()
    train_cfg.ablation = args.ablation
    if args.seed is not None:
        train_cfg.seed = args.seed
    env_cfg.seed = train_cfg.seed
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if train_cfg.runner.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    validate_config(env_cfg, train_cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_dir) if args.output_dir else Path(WHEEL_LEGGED_GYM_ROOT_DIR) / "logs" / "q1_tcn" / (
        "{}_{}_seed{}".format(stamp, args.ablation, train_cfg.seed))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output directory must be empty: {}".format(output))
    env, _ = task_registry.make_env("q1_tcn", args=args, env_cfg=env_cfg)
    try:
        runner = TCNPpoRunner(env, train_cfg, output, device=args.rl_device)
        if args.resume_checkpoint:
            runner.load(args.resume_checkpoint)
        # Snapshot the local network/algorithm and inherited flat control sources.
        source_dir = output / "source"
        source_dir.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob("*.py"):
            copy2(source, source_dir / source.name)
        for name in ("q1_flat.py", "q1_flat_config.py"):
            copy2(Path(__file__).parent.parent / "q1_flat" / name, source_dir / name)
        copy2(env_cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR), output / "q1.urdf")
        env.reset()
        # Resume restores optimizer/counters; the simulator starts fresh episodes.
        runner.learn(train_cfg.runner.max_iterations)
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
