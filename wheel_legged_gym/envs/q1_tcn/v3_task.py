"""Public train/play adapters around the frozen v3 environment and PPO runner."""
from datetime import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace


def checkpoint_path(args, root):
    from wheel_legged_gym.utils.helpers import get_load_path
    run = getattr(args, 'load_run', None)
    if run in (None, '-1'):
        run = -1
    experiment = getattr(args, 'experiment_name', None) or 'q1_tcn'
    directory = root/'logs'/experiment
    if isinstance(run, str) and Path(run).is_absolute():
        directory = Path(run).parent
    number = getattr(args, 'checkpoint', None)
    return Path(get_load_path(str(directory), load_run=run,
                             checkpoint=-1 if number is None else number)).resolve()


def training_config(args, checkpoint=None):
    from .q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO
    from .v3_profile import Q1TCNTaskCfg, Q1TCNTaskCfgPPO, validate_task_config
    from .runner import load_checkpoint, restore_config
    if checkpoint is None:
        cfg, train = Q1TCNTaskCfg(), Q1TCNTaskCfgPPO()
        if getattr(args, 'seed', None) is not None:
            cfg.seed = train.seed = args.seed
        if getattr(args, 'num_envs', None) is not None:
            cfg.env.num_envs = args.num_envs
    else:
        saved = load_checkpoint(checkpoint)
        if saved['config'].get('task') != 'q1_tcn':
            raise ValueError('Expected a q1_tcn checkpoint')
        cfg, train = Q1TCNCfg(), Q1TCNCfgPPO()
        restore_config(cfg, saved['config']['env'])
        restore_config(train, saved['config']['train'])
        for name, expected in [('seed', train.seed), ('num_envs', cfg.env.num_envs)]:
            value = getattr(args, name, None)
            if value is not None and value != expected:
                raise ValueError('Resume must preserve '+name)
    if getattr(args, 'max_iterations', None) is not None:
        train.runner.max_iterations = args.max_iterations
    if train.runner.max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if getattr(args, 'q1_stage', None) or getattr(args, 'ctbc_stage', None):
        raise ValueError('q1_tcn v3 does not use legacy stage flags')
    validate_task_config(cfg, train)
    return cfg, train


def train_task(args):
    import torch
    from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
    from wheel_legged_gym.utils import task_registry
    from .runner import TCNPpoRunner, config_dict
    from .v3_profile import PROFILE_VERSION
    root = Path(WHEEL_LEGGED_GYM_ROOT_DIR)
    checkpoint = checkpoint_path(args, root) if getattr(args, 'resume', False) else None
    cfg, train = training_config(args, checkpoint)
    args.seed, args.num_envs = train.seed, cfg.env.num_envs
    experiment = getattr(args, 'experiment_name', None) or 'q1_tcn'
    name = getattr(args, 'run_name', None) or PROFILE_VERSION
    output = root/'logs'/experiment/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+name)
    torch.set_num_threads(1)
    env, runner = None, None
    try:
        env, _ = task_registry.make_env('q1_tcn', args=args, env_cfg=cfg)
        runner = TCNPpoRunner(env, train, output, device=args.rl_device)
        if checkpoint is not None:
            runner.load(checkpoint)
        elif runner.iteration or runner.total_steps or runner.alg.optimizer.state:
            raise ValueError('Fresh training must have empty optimizer and zero counters')
        if train.runner.max_iterations <= runner.iteration:
            raise ValueError('Absolute iteration budget must exceed restored iteration')
        env.set_training_iteration(runner.iteration)
        env.reset()
        source = output/'source'
        source.mkdir()
        for path in Path(__file__).parent.iterdir():
            if path.suffix in ('.py', '.json'):
                shutil.copy2(path, source/path.name)
        shutil.copy2(root/'wheel_legged_gym/envs/q1_flat/q1_flat.py', source/'q1_flat.py')
        manifest = dict(profile_version=PROFILE_VERSION, checkpoint=str(checkpoint) if checkpoint else None,
                        initialization='resume' if checkpoint else 'random',
                        policy_and_adam_inherited=checkpoint is not None,
                        initial_iteration=runner.iteration, initial_total_steps=runner.total_steps,
                        teacher_weight=float(env.ctbc_feedforward_weight),
                        environment_state='Fresh simulator and curriculum; not stored in checkpoint',
                        budget=train.runner.max_iterations, environment=config_dict(cfg))
        (output/'initialization.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        runner.save()
        print('{} | {}'.format(PROFILE_VERSION, output), flush=True)
        runner.learn(train.runner.max_iterations-runner.iteration)
    except KeyboardInterrupt:
        if runner is not None:
            print('Interrupted; saved {}'.format(runner.save()), flush=True)
    finally:
        if env is not None:
            if env.viewer:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    return output


def play_task(args):
    from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
    from . import play
    path = checkpoint_path(args, Path(WHEEL_LEGGED_GYM_ROOT_DIR))
    options = SimpleNamespace(**vars(args))
    options.checkpoint_path = str(path)
    options.mode = getattr(args, 'play_mode', None) or 'keyboard'
    if options.mode == 'auto':
        options.mode = 'combined'
    options.duration = getattr(args, 'play_duration', None) or 600.
    options.tracking_speed, options.yaw_speed = .3, 1.
    return play.run(options)
