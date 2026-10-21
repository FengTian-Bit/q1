"""Record a saved Q1 checkpoint without creating a SwanLab experiment."""
import isaacgym  # Must load before torch.
import json
from pathlib import Path

import torch

from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry, class_to_dict
from wheel_legged_gym.rsl_rl.modules import ActorCriticSequence
from wheel_legged_gym.rsl_rl.utils.video import capture_video, video_settings


def apply_config(target, values):
    for key, value in values.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            apply_config(current, value)
        else:
            setattr(target, key, value)


def main():
    args = get_args(extra_custom_parameters=[
        {"name": "--checkpoint_path", "type": str, "required": True},
        {"name": "--video_output", "type": str, "required": True},
    ])
    if args.task != "q1":
        raise ValueError("Checkpoint recording currently supports --task=q1")
    checkpoint = Path(args.checkpoint_path).resolve()
    snapshot = json.loads((checkpoint.parent/"resolved_config.json").read_text(encoding="utf-8"))
    cfg, train_cfg = task_registry.get_cfgs("q1")
    apply_config(cfg, snapshot["env"])
    apply_config(train_cfg, snapshot["train"])
    stage = snapshot.get("stage", "turning")
    video_cfg = video_settings(class_to_dict(train_cfg.runner.periodic_video))
    cfg.env.num_envs = args.num_envs = 1
    cfg.env.enable_camera_sensors = True
    args.headless = True
    cfg.init_state.reset_lin_vel = cfg.init_state.reset_ang_vel = 0.0
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.stand_probability = 1.0
    cfg.terrain.curriculum = False
    terrain = video_cfg.get("terrain_type", "flat")
    if terrain not in cfg.terrain.terrain_names:
        raise ValueError("Unknown recording terrain: "+str(terrain))
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.selected = True
    cfg.terrain.selected_type = terrain
    cfg.terrain.selected_difficulty = float(video_cfg.get("terrain_difficulty", 0.25))
    if not 0 <= cfg.terrain.selected_difficulty <= 1:
        raise ValueError("Recording terrain difficulty must be in [0, 1]")
    if terrain == "flat":
        cfg.terrain.mesh_type = "plane"
    else:
        cfg.terrain.mesh_type = "trimesh"
    env = None
    try:
        env, _ = task_registry.make_env("q1", args=args, env_cfg=cfg)
        policy_cfg = class_to_dict(train_cfg.policy)
        model = ActorCriticSequence(env.num_obs, env.num_privileged_obs+policy_cfg["latent_dim"],
                                   env.num_actions, **policy_cfg).to(args.rl_device)
        model.load_state_dict(torch.load(str(checkpoint), map_location=args.rl_device)["model_state_dict"])
        model.eval()
        frames = capture_video(env, model.act_inference, args.video_output, video_cfg, stage)
        Path(args.video_output).with_suffix(".json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "stage": stage, "frames": frames,
            "terrain_type": terrain, "video": video_cfg,
        }, indent=2), encoding="utf-8")
        print("[training video] completed {} ({} frames)".format(args.video_output, frames), flush=True)
    finally:
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
