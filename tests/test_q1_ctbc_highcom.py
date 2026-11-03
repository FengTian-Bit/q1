"""High-COM geometry, actuator envelopes, and training/checkpoint integration.

The tensor Gym fixture exercises production methods but does not simulate
contacts or prove that the higher stance improves stair-climbing success.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_q1_flat import NS, ROOT, load_code, make_env
from test_q1_ctbc import NAMESPACE
from test_q1_ctbc_run_config import CONFIG, arguments, snapshot as original_snapshot


HIGH = dict(NAMESPACE, prepare_ctbc_training=CONFIG.prepare_ctbc_training)
load_code("wheel_legged_gym/envs/q1_ctbc_highcom/q1_ctbc_highcom_config.py", HIGH)
load_code("wheel_legged_gym/envs/q1_ctbc_highcom/q1_ctbc_highcom.py", HIGH)
load_code("wheel_legged_gym/envs/q1_ctbc_highcom/run_config.py", HIGH)

SPEC = importlib.util.spec_from_file_location("highcom_model_check", ROOT / "tools/check_q1_model.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def highcom_config(stage="single"):
    return HIGH["configure_highcom_stage"](HIGH["Q1CTBCHighCOMCfg"](), stage)


def highcom_env(stage="single", count=2, permutation=None):
    def nominal_config():
        cfg = highcom_config(stage)
        # These checks measure nominal IK; motor-zero randomization intentionally
        # changes the physical clearance and is covered by the base task tests.
        cfg.domain_rand.randomize_default_dof_pos = False
        return cfg

    env = make_env(count, permutation=permutation,
                   task_class=HIGH["Q1CTBCHighCOM"], cfg_class=nominal_config)
    props = np.zeros(env.num_dof, dtype=[(name, np.float32)
                                       for name in ("lower", "upper", "velocity", "effort")])
    props["lower"], props["upper"] = env.dof_pos_limits.T.numpy()
    props["velocity"], props["effort"] = env.dof_vel_limits.numpy(), env.torque_limits.numpy()
    env._process_dof_props(props, 0)
    env.external_commands = True
    env.commands[:, 0] = env.command_targets[:, 0] = 0.2
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    return env


def fk(q):
    return torch.stack((-.2*torch.sin(q[..., 0])-.2*torch.sin(q.sum(-1)),
                        -.2*torch.cos(q[..., 0])-.2*torch.cos(q.sum(-1))), dim=-1)


def highcom_snapshot(stage="single"):
    cfg = highcom_config(stage)
    train = HIGH["Q1CTBCHighCOMCfgPPO"]()
    return {"task": "q1_ctbc_highcom", "stage": stage,
            "env": NS["class_to_dict"](cfg), "train": NS["class_to_dict"](train)}


class Q1CTBCHighCOMTests(unittest.TestCase):
    def test_full_urdf_com_alignment_ground_clearance_and_inertia_screen(self):
        cfg = highcom_config()
        report = MODEL.check(ROOT / "resources/robots/q1/urdf/q1.urdf",
                             initial_angles=cfg.init_state.default_joint_angles,
                             initial_height=cfg.init_state.pos[2])
        self.assertEqual(report["errors"], [])
        self.assertEqual((report["link_count"], report["active_joint_count"]), (9, 6))
        self.assertAlmostEqual(report["total_mass_kg"], 6.87031059)
        self.assertAlmostEqual(cfg.commands.ranges.height[0], .4360150397, places=8)
        wheels = [row["initial_bottom_m"] for row in report["collisions"] if row["link"].endswith("_foot")]
        np.testing.assert_allclose(wheels, [.00534, .00534], atol=1e-9)
        self.assertTrue(all(row["initial_bottom_m"] >= 0 for row in report["collisions"]))
        q = torch.tensor([.3746, -.7854], dtype=torch.float64)
        wheel_x = -.025225 + fk(q)[0].item()
        self.assertLess(abs(report["initial_total_com_m"][0]-wheel_x), .0001)
        # COM height is distinct from the configured trunk-origin height.
        self.assertAlmostEqual(report["initial_total_com_m"][2]-.00534, .3246758811, places=8)

    def test_all_stages_keep_the_new_pose_without_changing_the_original_task(self):
        original = NAMESPACE["Q1CTBCCfg"]()
        baseline = NS["class_to_dict"](original)
        for stage in ("single", "stairs", "robust", "stairs20"):
            cfg = highcom_config(stage)
            for side in ("L", "R"):
                self.assertEqual(cfg.init_state.default_joint_angles[side+"_thigh_joint"], .3746)
                self.assertEqual(cfg.init_state.default_joint_angles[side+"_calf_joint"], -.7854)
            self.assertAlmostEqual(cfg.init_state.pos[2]-cfg.commands.ranges.height[0], .00534)
            self.assertEqual(cfg.rewards.base_height_target, cfg.commands.ranges.height[0])
            self.assertEqual(cfg.ctbc.stage, stage)
            self.assertEqual(cfg.control.stiffness, original.control.stiffness)
            self.assertEqual(cfg.control.damping, original.control.damping)
            self.assertEqual(cfg.control.leg_action_scales, [.5, .9, .5, .9])
        self.assertEqual(NS["class_to_dict"](original), baseline)
        self.assertEqual(NS["class_to_dict"](NAMESPACE["Q1CTBCCfg"]()), baseline)

    def test_low_stage_teacher_retains_eight_cm_lift_and_wheel_x(self):
        for stage in ("single", "stairs", "robust"):
            with self.subTest(stage=stage):
                env = highcom_env(stage)
                env.set_guidance_mode("on")
                env.guidance.active_leg[:] = 0
                env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration/2-env.dt
                env.step(torch.zeros(2, 6))
                initial = fk(env.default_dof_pos[:, env.leg_dof_indices[:2]])
                target = fk(env.position_target[:, env.leg_dof_indices[:2]])
                torch.testing.assert_close(target-initial, torch.tensor([[0., .08]]).repeat(2, 1),
                                           atol=1e-6, rtol=0)
                self.assertTrue(torch.isfinite(env.obs_buf).all())
                self.assertEqual((env.obs_buf.shape[1], env.privileged_obs_buf.shape[1],
                                  env.obs_history.shape[1]), (31, 151, 155))

    def test_high_stage_guide_keeps_eight_to_sixteen_cm_across_all_rows(self):
        env = highcom_env("stairs20", count=6)
        env.ctbc_riser_heights[:] = torch.linspace(.10, .20, 6)
        env._update_guide_amplitudes()
        env.set_guidance_mode("on")
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = env.cfg.ctbc.swing_duration/2-env.dt
        env.step(torch.zeros(6, 6))
        initial = fk(env.default_dof_pos[:, env.leg_dof_indices[:2]])
        target = fk(env.position_target[:, env.leg_dof_indices[:2]])
        torch.testing.assert_close(target[:, 0], initial[:, 0], atol=1e-6, rtol=0)
        torch.testing.assert_close(target[:, 1]-initial[:, 1], torch.linspace(.08, .16, 6),
                                   atol=1e-6, rtol=0)

    def test_pure_policy_twenty_cm_lift_is_not_clipped_and_real_motor_limits_hold(self):
        env = highcom_env("stairs20", permutation=[5, 2, 4, 0, 3, 1])
        env.set_guidance_mode("off")
        actions = torch.zeros(2, 6)
        actions[:, :2] = torch.tensor(HIGH["nominal_lift_offsets"](.20))/torch.tensor([.5, .9])
        raw = actions.clone()
        env.step(actions)
        initial = fk(env.default_dof_pos[:, env.leg_dof_indices[:2]])
        target = fk(env.position_target[:, env.leg_dof_indices[:2]])
        torch.testing.assert_close(target-initial, torch.tensor([[0., .20]]).repeat(2, 1),
                                   atol=1e-6, rtol=0)
        torch.testing.assert_close(actions, raw)
        self.assertEqual(env.cfg.control.leg_action_limits, [.8, 1.55, .8, 1.55])
        self.assertTrue((env.torques.abs() <= env.torque_limits).all())
        torch.testing.assert_close(env.torque_limits[env.policy_dof_indices],
                                   torch.tensor([17., 17., 6., 17., 17., 6.]))
        limits = env.dof_pos_limits[env.leg_dof_indices[:2]]
        self.assertTrue((env.position_target[:, env.leg_dof_indices[:2]] >= limits[:, 0]+.05).all())

    def test_saved_task_and_posture_profile_prevent_cross_task_resume(self):
        data = highcom_snapshot()
        self.assertEqual(CONFIG.validate_ctbc_snapshot(data, task_name="q1_ctbc_highcom"), "single")
        with self.assertRaisesRegex(ValueError, "saved task"):
            CONFIG.validate_ctbc_snapshot(data)
        with self.assertRaisesRegex(ValueError, "saved task"):
            CONFIG.validate_ctbc_snapshot(original_snapshot(), task_name="q1_ctbc_highcom")
        del data["env"]["ctbc"]["stance_profile"]
        with self.assertRaisesRegex(ValueError, "stance profile"):
            CONFIG.validate_ctbc_snapshot(data, task_name="q1_ctbc_highcom")

    def test_resume_preserves_highcom_config_and_stage_entry_guidance_clock(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            run = Path(folder)
            checkpoint = run / "model_12000.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            data = highcom_snapshot()
            data["env"]["ctbc"]["feedforward_hold_iterations"] = 4321
            (run / "resolved_config.json").write_text(json.dumps(data), encoding="utf-8")
            cfg, train = highcom_config(), HIGH["Q1CTBCHighCOMCfgPPO"]()
            args = arguments(ctbc_stage="stairs20")
            resolver = Mock(return_value=str(checkpoint))
            with patch.object(CONFIG, "_load_checkpoint_iteration", return_value=12000):
                HIGH["prepare_highcom_training"](cfg, train, args, folder, resolver)
            self.assertAlmostEqual(cfg.commands.ranges.height[0], .4360150397, places=8)
            self.assertEqual(cfg.control.leg_action_limits, [.8, 1.55, .8, 1.55])
            self.assertEqual(cfg.ctbc.feedforward_hold_iterations, 4321)
            self.assertEqual(cfg.ctbc.feedforward_start_iteration, 12000)
            self.assertEqual(train.runner.experiment_name, "q1_ctbc_highcom")
            self.assertEqual(args.load_run, str(run.resolve()))

    def test_train_entry_saves_highcom_snapshot_sources_and_disables_random_episode_clock(self):
        cfg, train = highcom_config(), HIGH["Q1CTBCHighCOMCfgPPO"]()
        learn = Mock()
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            log_dir = Path(folder) / "run"
            registry = SimpleNamespace(
                log_dir=str(log_dir), get_cfgs=lambda name: (cfg, train),
                make_env=lambda **kw: (object(), cfg),
                make_alg_runner=lambda **kw: (SimpleNamespace(learn=learn), train),
                save_cfgs=lambda **kw: log_dir.mkdir())
            namespace = dict(task_registry=registry, os=os, WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT))
            load_code("wheel_legged_gym/scripts/train.py", namespace)
            modules = {
                "wheel_legged_gym.envs.q1_ctbc_highcom.run_config": SimpleNamespace(
                    prepare_highcom_training=HIGH["prepare_highcom_training"]),
                "wheel_legged_gym.utils.helpers": SimpleNamespace(
                    get_load_path=Mock(), class_to_dict=NS["class_to_dict"])}
            args = arguments(resume=False, task="q1_ctbc_highcom", ctbc_stage="stairs20")
            with patch.dict(sys.modules, modules):
                namespace["train"](args)
            saved = json.loads((log_dir / "resolved_config.json").read_text())
            self.assertEqual(saved["task"], "q1_ctbc_highcom")
            self.assertEqual(saved["stage"], "stairs20")
            self.assertEqual(saved["env"]["init_state"]["default_joint_angles"]["L_thigh_joint"], .3746)
            self.assertTrue((log_dir / "source_highcom_run_config.py").is_file())
            self.assertTrue((log_dir / "source_q1_ctbc.py").is_file())
            learn.assert_called_once_with(num_learning_iterations=train.runner.max_iterations,
                                          init_at_random_ep_len=False)


if __name__ == "__main__":
    unittest.main()
