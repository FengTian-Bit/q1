"""CPU behaviour checks for contact instructions; no physics claims are made."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PATH = (Path(__file__).resolve().parents[1] / "wheel_legged_gym" / "envs"
        / "q1_ctbc" / "guidance.py")
SPEC = importlib.util.spec_from_file_location("q1_ctbc_guidance", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ContactGuidance = MODULE.ContactGuidance


def make_guidance(count=1, **overrides):
    settings = dict(contact_threshold=12.0, contact_frames=3,
                    swing_duration=0.6, min_swing_time=0.3, cooldown_s=0.2,
                    hip_amplitude=0.2952, knee_amplitude=-0.5905,
                    feedforward_initial=1.0, feedforward_hold_iterations=3000,
                    feedforward_anneal_iterations=12000,
                    trigger_enabled=True, feedforward_enabled=True)
    settings.update(overrides)
    return ContactGuidance(count, "cpu", 0.01, SimpleNamespace(**settings))


def sample(guidance, forces, support=None, moving=None, height=None, steps=1):
    forces = torch.tensor(forces, dtype=torch.float32).reshape(guidance.num_envs, 2)
    support = (torch.ones_like(forces, dtype=torch.bool) if support is None
               else torch.tensor(support, dtype=torch.bool).reshape_as(forces))
    moving = (torch.ones(guidance.num_envs, dtype=torch.bool) if moving is None
              else torch.tensor(moving, dtype=torch.bool))
    height = torch.zeros_like(forces) if height is None else torch.tensor(height)
    for _ in range(steps):
        guidance.update(forces, support, height, moving)


class ContactGuidanceTests(unittest.TestCase):
    def test_three_consecutive_frames_and_threshold(self):
        guide = make_guidance()
        sample(guide, [[13, 0]], steps=2)
        self.assertEqual(guide.active_leg.item(), -1)
        sample(guide, [[12, 0]])  # Equality does not exceed the threshold.
        sample(guide, [[13, 0]], steps=2)
        self.assertEqual(guide.active_leg.item(), -1)
        sample(guide, [[13, 0]])
        self.assertEqual(guide.active_leg.item(), 0)
        self.assertEqual(guide.triggered.tolist(), [[True, False]])

    def test_stable_single_side_wins_over_transient_larger_force(self):
        guide = make_guidance()
        sample(guide, [[13, 0]], steps=2)
        sample(guide, [[13, 99]])
        self.assertEqual(guide.active_leg.item(), 0)

    def test_simultaneous_collision_selects_larger_and_enforces_exclusion(self):
        guide = make_guidance(3)
        sample(guide, [[14, 20], [30, 15], [16, 16]], steps=3)
        self.assertEqual(guide.active_leg.tolist(), [1, 0, 0])
        self.assertTrue(torch.all(guide.swing_mask.sum(1) == 1))
        sample(guide, [[100, 0], [0, 100], [0, 100]], steps=5)
        self.assertEqual(guide.active_leg.tolist(), [1, 0, 0])
        self.assertFalse(guide.triggered.any())

    def test_trigger_latches_after_force_disappears_and_while_stopping(self):
        guide = make_guidance()
        sample(guide, [[0, 15]], height=[[0.01, 0.03]], steps=3)
        sample(guide, [[0, 0]], moving=[False], steps=10)
        self.assertEqual(guide.active_leg.item(), 1)
        self.assertAlmostEqual(guide.elapsed.item(), 0.1, places=6)
        torch.testing.assert_close(guide.start_height, torch.tensor([[0.01, 0.03]]))

    def test_stationary_and_disabled_do_not_accumulate_triggers(self):
        guide = make_guidance(2)
        sample(guide, [[50, 50], [50, 50]], moving=[False, True], steps=3)
        self.assertEqual(guide.active_leg.tolist(), [-1, 0])
        disabled = make_guidance(trigger_enabled=False)
        sample(disabled, [[50, 50]], steps=10)
        self.assertEqual(disabled.active_leg.item(), -1)

    def test_initial_support_cannot_finish_unexecuted_swing(self):
        guide = make_guidance()
        sample(guide, [[20, 0]], steps=3)
        sample(guide, [[0, 0]], steps=35)
        self.assertEqual(guide.active_leg.item(), 0)
        self.assertFalse(guide.has_unloaded.item())
        sample(guide, [[0, 0]], support=[[False, True]])
        sample(guide, [[0, 0]])
        self.assertEqual(guide.active_leg.item(), -1)
        self.assertFalse(guide.timed_out.item())

    def test_landing_waits_for_minimum_time_and_cooldown(self):
        guide = make_guidance()
        sample(guide, [[20, 0]], steps=3)
        sample(guide, [[0, 0]], support=[[False, True]], steps=10)
        sample(guide, [[0, 0]], steps=19)
        self.assertEqual(guide.active_leg.item(), 0)
        sample(guide, [[20, 0]])
        self.assertEqual(guide.active_leg.item(), -1)
        self.assertGreater(guide.cooldown.item(), 0)
        sample(guide, [[20, 0]], steps=19)
        self.assertEqual(guide.active_leg.item(), -1)
        sample(guide, [[20, 0]], steps=3)
        self.assertEqual(guide.active_leg.item(), 0)

    def test_timeout_is_one_step_event_and_clears_instruction(self):
        guide = make_guidance()
        sample(guide, [[20, 0]], steps=3)
        sample(guide, [[0, 0]], steps=59)
        self.assertEqual(guide.active_leg.item(), 0)
        sample(guide, [[0, 0]])
        self.assertEqual(guide.active_leg.item(), -1)
        self.assertTrue(guide.timed_out.item())
        self.assertFalse(guide.reference_offsets().any())
        sample(guide, [[0, 0]])
        self.assertFalse(guide.timed_out.item())

    def test_reset_isolates_neighbouring_swing_and_partial_debounce(self):
        guide = make_guidance(3)
        sample(guide, [[20, 0], [0, 20], [0, 0]], steps=3)
        sample(guide, [[0, 0], [0, 0], [20, 0]], steps=2)
        before = guide.elapsed[1].clone()
        guide.reset(torch.tensor([0, 2]))
        self.assertEqual(guide.active_leg.tolist(), [-1, 1, -1])
        torch.testing.assert_close(guide.elapsed[1], before)
        self.assertFalse(guide.start_height[[0, 2]].any())
        sample(guide, [[0, 0], [0, 0], [20, 0]])
        self.assertEqual(guide.active_leg.tolist(), [-1, 1, -1])
        sample(guide, [[0, 0], [0, 0], [20, 0]], steps=2)
        self.assertEqual(guide.active_leg.tolist(), [-1, 1, 0])

    def test_cosine_peak_and_joint_order(self):
        guide = make_guidance(2)
        sample(guide, [[20, 0], [0, 20]], steps=3)
        torch.testing.assert_close(guide.reference_offsets(), torch.zeros(2, 4))
        sample(guide, [[0, 0], [0, 0]], steps=30)
        torch.testing.assert_close(guide.phase, torch.full((2,), 0.5))
        torch.testing.assert_close(guide.reference_offsets(), torch.tensor([
            [0.2952, -0.5905, 0.0, 0.0], [0.0, 0.0, 0.2952, -0.5905]]))

    def test_feedforward_holds_then_linearly_reaches_zero(self):
        guide = make_guidance()
        for iteration in (0, 2999, 3000):
            self.assertEqual(guide.feedforward_weight(iteration), 1.0)
        self.assertEqual(guide.feedforward_weight(9000), 0.5)
        self.assertEqual(guide.feedforward_weight(15000), 0.0)
        self.assertEqual(guide.feedforward_weight(20000), 0.0)
        guide.cfg.feedforward_enabled = False
        self.assertEqual(guide.feedforward_weight(0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
