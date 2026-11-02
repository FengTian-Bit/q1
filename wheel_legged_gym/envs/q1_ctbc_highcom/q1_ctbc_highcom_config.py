"""A 45-degree knee stance for an isolated Q1 CTBC posture experiment.

The small thigh offset from half the knee angle aligns the full URDF COM
with the wheel axle at zero trunk pitch. Heights refer to the trunk origin,
not the whole-robot COM. Motor limits, gains and normalized action scales
are inherited unchanged from q1_ctbc.
"""
import math

from wheel_legged_gym.envs.q1_ctbc.q1_ctbc_config import (
    Q1CTBCCfg, Q1CTBCCfgPPO, configure_ctbc_stage,
)


HIGHCOM_THIGH = 0.3746
HIGHCOM_CALF = -0.7854
HIGHCOM_HEIGHT = (0.2*math.cos(HIGHCOM_THIGH)
                  + 0.2*math.cos(HIGHCOM_THIGH+HIGHCOM_CALF)
                  + 0.07496 - 0.0084361)
HIGHCOM_SPAWN_CLEARANCE = 0.00534


def nominal_lift_offsets(lift):
    """Two-link IK retaining the nominal wheel X coordinate, in radians."""
    x = -0.2*math.sin(HIGHCOM_THIGH)-0.2*math.sin(HIGHCOM_THIGH+HIGHCOM_CALF)
    z = -0.2*math.cos(HIGHCOM_THIGH)-0.2*math.cos(HIGHCOM_THIGH+HIGHCOM_CALF)+lift
    knee = -math.acos((x*x+z*z-0.08)/0.08)
    thigh = math.atan2(-x, -z)-math.atan2(0.2*math.sin(knee), 0.2+0.2*math.cos(knee))
    return thigh-HIGHCOM_THIGH, knee-HIGHCOM_CALF


def configure_highcom_stage(cfg, stage):
    """Apply the matching CTBC curriculum and then its posture dependencies."""
    configure_ctbc_stage(cfg, stage, high_stairs_profile="legacy20")
    cfg.init_state.default_joint_angles = {
        side+"_"+joint+"_joint": angle for side in ("L", "R")
        for joint, angle in (("thigh", HIGHCOM_THIGH), ("calf", HIGHCOM_CALF), ("foot", 0.0))}
    cfg.init_state.pos = [0.0, 0.0, HIGHCOM_HEIGHT+HIGHCOM_SPAWN_CLEARANCE]
    cfg.commands.ranges.height = [HIGHCOM_HEIGHT, HIGHCOM_HEIGHT]
    cfg.rewards.base_height_target = HIGHCOM_HEIGHT
    # The old fixed angular pulse no longer produces an 8 cm lift from this
    # stance. Use the existing IK path for every stage, retaining physical lift.
    cfg.ctbc.guide_lift_height_range = [0.08, 0.16] if stage == "stairs20" else [0.08, 0.08]
    cfg.ctbc.hip_amplitude, cfg.ctbc.knee_amplitude = nominal_lift_offsets(0.08)
    cfg.ctbc.stance_profile = "highcom_knee45_v1"
    if stage == "stairs20":
        # From the straighter nominal stance, 20 cm clearance requires larger
        # offsets even though the final knee angle is farther from its limit.
        # Preserve rad/action and enlarge only the high-stage offset envelope.
        cfg.control.leg_action_limits = [0.8, 1.55, 0.8, 1.55]
        cfg.normalization.clip_actions = max(
            [1.0]+[limit/scale for limit, scale in zip(
                cfg.control.leg_action_limits, cfg.control.leg_action_scales)])
    return cfg


class Q1CTBCHighCOMCfg(Q1CTBCCfg):
    class env(Q1CTBCCfg.env):
        # Keep this separate posture experiment's existing training batch.
        num_envs = 4096

    def __init__(self):
        super().__init__()
        configure_highcom_stage(self, "single")


class Q1CTBCHighCOMCfgPPO(Q1CTBCCfgPPO):
    class runner(Q1CTBCCfgPPO.runner):
        experiment_name = "q1_ctbc_highcom"
        run_name = "single"
        swanlab_tags = ["q1", "ctbc", "highcom-knee45"]
