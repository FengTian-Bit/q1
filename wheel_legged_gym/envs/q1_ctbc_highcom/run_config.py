"""Restore high-COM runs without accepting a different action-centre posture."""
from wheel_legged_gym.envs.q1_ctbc.run_config import prepare_ctbc_training
from .q1_ctbc_highcom_config import configure_highcom_stage


def prepare_highcom_training(env_cfg, train_cfg, args, root_dir, get_load_path):
    return prepare_ctbc_training(
        env_cfg, train_cfg, args, root_dir, get_load_path,
        task_name="q1_ctbc_highcom", stage_configurator=configure_highcom_stage)
