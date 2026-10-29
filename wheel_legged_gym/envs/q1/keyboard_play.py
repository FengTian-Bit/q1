"""Explicit small-environment terrain gallery for Q1 manual evaluation."""
from copy import deepcopy
import math


# Gym pyramids spawn at their centres. Negative stairs form a pit, so driving
# out climbs; positive stairs form a hill, so driving out descends.
GALLERY = ("stairs_down", "stairs_up", "slope_up", "slope_down", "rough_slope", "flat", "discrete")


def configure_keyboard_play(cfg, args):
    """Copy configs so a playback override cannot alter registered training configs."""
    if args.task != "q1":
        raise ValueError("keyboard_play currently supports --task=q1")
    if args.headless or args.play_mode != "keyboard":
        raise ValueError("keyboard_play requires a viewer and --play_mode=keyboard")
    count = 6 if args.num_envs is None else args.num_envs
    if count < 1:
        raise ValueError("--num_envs must be positive")
    if not math.isfinite(args.difficulty) or not 0 <= args.difficulty <= 1:
        raise ValueError("--difficulty must be in [0, 1]")
    if args.terrain != "mixed" and args.terrain not in GALLERY:
        raise ValueError("Unsupported playback terrain: " + args.terrain)
    cfg = deepcopy(cfg)
    cfg.env.num_envs = count
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.curriculum = False
    cfg.terrain.selected = False
    cfg.terrain.num_rows, cfg.terrain.num_cols = 1, count
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.playback_layout = [GALLERY[i % len(GALLERY)] if args.terrain == "mixed"
                                   else args.terrain for i in range(count)]
    cfg.terrain.playback_difficulty = args.difficulty
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if key.startswith("randomize_") or key == "push_robots":
            setattr(cfg.domain_rand, key, False)
    args.num_envs = count
    print_terrain_layout(cfg)
    return cfg


def print_terrain_layout(cfg):
    print("Q1 keyboard terrain gallery (fixed difficulty={:.2f}; curriculum OFF):".format(
        cfg.terrain.playback_difficulty))
    height = cfg.terrain.stair_height_range
    nominal = height[0] + (height[1]-height[0])*cfg.terrain.playback_difficulty
    quantized = int(nominal/cfg.terrain.vertical_scale)*cfg.terrain.vertical_scale
    for robot, name in enumerate(cfg.terrain.playback_layout):
        note = ""
        if name in ("stairs_down", "stairs_up"):
            note = "; outward: {} stairs; riser ~{:.1f} cm".format(
                "UP" if name == "stairs_down" else "DOWN", quantized*100)
        print("  robot {}: {}{}".format(robot, name, note))
    print("Spawn platform is {:.1f} m wide: drive about {:.1f} m to reach stairs. "
          "Robot 0 is selected; W/S drive, A/D turn, X stop; 0-8 or [/]: select.".format(
              cfg.terrain.platform_size, cfg.terrain.platform_size/2))


def keyboard_play(args):
    from wheel_legged_gym.envs.q1.play_t0 import play_q1
    return play_q1(args, playback_configure=configure_keyboard_play)
