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

    def test_amplitudes_are_environment_specific_and_preserved_across_reset(self):
        guide = make_guidance(3)
        changed = torch.tensor([[0.53146, -1.06293], [0.4, -0.8]])
        guide.set_amplitudes(torch.tensor([1, 2]), changed)
        sample(guide, [[20, 0], [0, 20], [20, 0]], steps=3)
        sample(guide, [[0, 0], [0, 0], [0, 0]], steps=30)
        torch.testing.assert_close(guide.reference_offsets(), torch.tensor([
            [0.2952, -0.5905, 0.0, 0.0],
            [0.0, 0.0, 0.53146, -1.06293],
            [0.4, -0.8, 0.0, 0.0]]))
        guide.reset(torch.tensor([1]))
        torch.testing.assert_close(guide.amplitudes[1:], changed)
        self.assertEqual(guide.active_leg.tolist(), [0, -1, 0])
        torch.testing.assert_close(guide.reference_offsets()[1], torch.zeros(4))

    def test_invalid_amplitude_updates_are_rejected_atomically(self):
        guide = make_guidance(2)
        before = guide.amplitudes.clone()
        for values in (torch.tensor([[float("nan"), -0.5]]),
                       torch.tensor([[0.2, float("inf")]]),
                       torch.zeros(2), torch.zeros(1, 3), torch.zeros(2, 2)):
            with self.assertRaises(ValueError):
                guide.set_amplitudes(torch.tensor([1]), values)
            torch.testing.assert_close(guide.amplitudes, before)
        with self.assertRaises(ValueError):
            make_guidance(hip_amplitude=float("nan"))

    def test_feedforward_schedule_uses_relative_iterations_after_restart(self):
        guide = make_guidance(feedforward_start_iteration=100,
                              feedforward_initial=0.75,
                              feedforward_hold_iterations=10,
                              feedforward_anneal_iterations=20)
        # Before the offset, the configured initial value remains available;
        # entering a later stage does not consume its hold period in advance.
        for iteration in (0, 99, 100, 109, 110):
            self.assertEqual(guide.feedforward_weight(iteration), 0.75)
        self.assertEqual(guide.feedforward_weight(120), 0.375)
        self.assertEqual(guide.feedforward_weight(130), 0.0)
        self.assertEqual(guide.feedforward_weight(200), 0.0)
        guide.cfg.feedforward_anneal_iterations = 0
        self.assertEqual(guide.feedforward_weight(109), 0.75)
        self.assertEqual(guide.feedforward_weight(110), 0.0)

    def test_restart_ramp_is_part_of_hold_before_annealing(self):
        guide = make_guidance(feedforward_start_iteration=30000,
                              feedforward_ramp_iterations=1000,
                              feedforward_hold_iterations=3000,
                              feedforward_anneal_iterations=12000)
        for iteration in (0, 29999, 30000):
            self.assertEqual(guide.feedforward_weight(iteration), 0.0)
        self.assertEqual(guide.feedforward_weight(30500), 0.5)
        self.assertEqual(guide.feedforward_weight(31000), 1.0)
        self.assertEqual(guide.feedforward_weight(32999), 1.0)
        self.assertEqual(guide.feedforward_weight(33000), 1.0)
        self.assertEqual(guide.feedforward_weight(39000), 0.5)
        self.assertEqual(guide.feedforward_weight(45000), 0.0)
        guide.cfg.feedforward_enabled = False
        self.assertEqual(guide.feedforward_weight(31000), 0.0)

    def test_ramp_outside_hold_is_rejected(self):
        for ramp in (-1, 3001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_guidance(feedforward_ramp_iterations=ramp)
        guide = make_guidance()
        guide.cfg.feedforward_ramp_iterations = 3001
        with self.assertRaises(ValueError):
            guide.feedforward_weight(0)

    def test_landing_rise_uses_the_active_wheel_not_the_other_support(self):
        guide = make_guidance(min_landing_rise=0.02)
        sample(guide, [[20, 0]], height=[[0.03, 0.0]], steps=3)
        sample(guide, [[0, 0]], support=[[False, True]],
               height=[[0.04, 0.0]], steps=10)
        # The right wheel being high cannot qualify a left-wheel landing.
        sample(guide, [[0, 0]], height=[[0.04, 0.5]], steps=25)
        self.assertEqual(guide.active_leg.item(), 0)
        self.assertTrue(guide.has_unloaded.item())
        sample(guide, [[0, 0]], height=[[0.05, 0.0]])
        self.assertEqual(guide.active_leg.item(), -1)
        self.assertFalse(guide.timed_out.item())

    def test_insufficient_landing_rise_still_ends_at_timeout(self):
        guide = make_guidance(min_landing_rise=0.02)
        sample(guide, [[0, 20]], steps=3)
        sample(guide, [[0, 0]], support=[[True, False]], steps=10)
        sample(guide, [[0, 0]], height=[[0.0, 0.01]], steps=50)
        self.assertEqual(guide.active_leg.item(), -1)
        self.assertTrue(guide.timed_out.item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
