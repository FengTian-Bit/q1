"""Q1 keyboard playback, borrowing Dreamwaq's viewer controls and evaluation setup."""
import isaacgym  # Isaac Gym must load before torch.
from wheel_legged_gym.envs import *  # Register tasks.
from wheel_legged_gym.utils import get_args
from wheel_legged_gym.envs.q1.keyboard_play import GALLERY, keyboard_play


def main():
    args = get_args(default_task="q1", extra_custom_parameters=[
        {"name": "--terrain", "type": str, "default": "mixed",
         "choices": ["mixed"] + list(GALLERY),
         "help": "Explicit terrain per robot; mixed starts with both stair directions."},
        {"name": "--difficulty", "type": float, "default": 0.5,
         "help": "Fixed terrain difficulty in [0, 1]; 1 gives 20 cm stair risers."},
    ])
    keyboard_play(args)


if __name__ == "__main__":
    main()
