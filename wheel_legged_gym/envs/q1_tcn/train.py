"""Train the registered v3 profile with the normal legged_gym arguments."""
import isaacgym  # Must precede torch.
import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args
from .task import train_task

if __name__ == "__main__":
    train_task(get_args(default_task="q1_tcn"))
