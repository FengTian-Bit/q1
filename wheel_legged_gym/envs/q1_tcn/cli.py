"""Parse task options separately from version-dependent Isaac Gym arguments."""

import argparse
import sys


SCENARIOS = ("stand", "tracking", "push_front", "push_back", "push_side", "low_friction", "payload")


def task_parser(evaluation=False):
    parser = argparse.ArgumentParser(add_help=False, description="Q1 TCN task options")
    parser.add_argument("--output_dir", type=str, default=None)
    if evaluation:
        parser.add_argument("--checkpoint_path", type=str, required=True)
        parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
        parser.add_argument("--eval_seeds", nargs="+", type=int, default=[101, 102, 103])
        parser.add_argument("--duration", type=float, default=12.0)
        parser.add_argument("--push_force", type=float, default=8.0)
    else:
        parser.add_argument("--ablation", choices=["A0", "A1", "A2", "A3", "A4"], default="A1")
        parser.add_argument("--resume_checkpoint", type=str, default=None)
    return parser


def get_tcn_args(get_gym_args, evaluation=False):
    parser = task_parser(evaluation)
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        # Let Gym also print its simulator/device options.
        get_gym_args(default_task="q1_tcn")
    task_args, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        args = get_gym_args(default_task="q1_tcn")
    finally:
        sys.argv = original
    for name, value in vars(task_args).items():
        setattr(args, name, value)
    return args
