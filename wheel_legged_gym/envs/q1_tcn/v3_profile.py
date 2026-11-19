"""Registered r11r11000 settings; JSON holds configuration, never weights."""
import json
from pathlib import Path

from .q1_tcn_config import Q1TCNCfg, Q1TCNCfgPPO, validate_config
from .runner import restore_config

PROFILE_VERSION = 'q1_tcn_v3_r11r11000'
CONFIG_PATH = Path(__file__).with_name('v3_config.json')


def profile_config():
    with CONFIG_PATH.open(encoding='utf-8') as stream:
        return json.load(stream)


class Q1TCNTaskCfg(Q1TCNCfg):
    def __init__(self):
        super().__init__()
        restore_config(self, profile_config()['env'])


class Q1TCNTaskCfgPPO(Q1TCNCfgPPO):
    def __init__(self):
        super().__init__()
        restore_config(self, profile_config()['train'])


def validate_task_config(cfg, train, training=True):
    validate_config(cfg, train)
    if cfg.env.num_observations != 30 or cfg.env.obs_history_length != 61:
        raise ValueError('Q1 TCN v3 requires30 observations and61 stored frames')
    if train.ablation != 'A0' or not train.policy.slow_branch:
        raise ValueError('Q1 TCN v3 requires the direct A0 actor and slow TCN')
    if not train.policy.slope_head or not train.policy.clearance_head:
        raise ValueError('Q1 TCN v3 requires slope and clearance features')
    if train.policy.parking_policy or train.policy.parking_blend:
        raise ValueError('Q1 TCN v3 uses one terrain actor without a parking branch')
