"""Use the existing CTBC lifecycle with the high-COM task's own configuration."""

from wheel_legged_gym.envs.q1_ctbc.q1_ctbc import Q1CTBC


class Q1CTBCHighCOM(Q1CTBC):
    """Keep Q1's six-action controller, contact teacher and termination rules."""
