"""Exercise the actual CTBC environment lifecycle with a tensor Gym stub.

These tests validate instructions, observability and episode accounting. They
do not simulate contact dynamics or establish physical stair-climbing success.
"""

from collections import deque
import os
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
from torch import optim

from test_q1_flat import NS, load_code, make_env
from test_q1_ctbc_guidance import ContactGuidance


NAMESPACE = dict(NS, ContactGuidance=ContactGuidance)
load_code("wheel_legged_gym/envs/q1_ctbc/q1_ctbc_config.py", NAMESPACE)
load_code("wheel_legged_gym/envs/q1_ctbc/q1_ctbc.py", NAMESPACE)


def ctbc_env(count=2, permutation=None, max_delay_ms=0):
    env = make_env(count, permutation=permutation, max_delay_ms=max_delay_ms,
                   task_class=NAMESPACE["Q1CTBC"],
                   cfg_class=NAMESPACE["Q1CTBCCfg"])
    env.external_commands = True
    env.commands[:, 0] = 0.2
    env.command_targets[:, 0] = 0.2
    env.contact_forces[:, env.feet_indices, 2] = 40.0
    return env


def place_at_goal(env):
    env.ctbc_riser_heights[:] = 0.05
    env.ctbc_step_counts[:] = 1
    env.ctbc_goal_x[:] = 0.9
    env.root_states[:, 0] = 1.0
    env.root_states[:, 2] = 0.447
    env.rigid_body_states[:, env.feet_indices, 0] = 1.0
    env.rigid_body_states[:, env.feet_indices, 2] = 0.05 + env.cfg.asset.wheel_radius
    env.contact_forces[:, env.feet_indices, 2] = 40.0


class Q1CTBCTests(unittest.TestCase):
    def test_symmetry_is_enabled_and_gated_by_either_wheels_horizontal_force(self):
        env = ctbc_env(6, permutation=[5, 2, 4, 0, 3, 1])
        env._flat_terrain_mask = lambda: torch.tensor([True, False, False, False, False, False])
        env.dof_pos[:, env.leg_dof_indices[0]] += 0.3
        self.assertEqual(env.cfg.rewards.rough_leg_symmetry_weight, 1.0)
        self.assertEqual(env.cfg.rewards.scales.leg_symmetry, -0.10)
        baseline = NS["Q1"]._reward_leg_symmetry(env)
        self.assertTrue((baseline > 0).all())
        self.assertIn("leg_symmetry", env.reward_names)
        self.assertLess(env.reward_scales["leg_symmetry"], 0.0)

        # Use a non-default threshold to verify the existing setting is reused.
        env.cfg.ctbc.contact_threshold = 10.0
        env.contact_forces[:, env.feet_indices, 2] = 1000.0
        env.contact_forces[0, env.feet_indices[0], 0] = 11.0
        env.contact_forces[1, env.feet_indices[1], 1] = -11.0
        env.contact_forces[2, env.feet_indices[0], :2] = torch.tensor([8.0, 8.0])
        env.contact_forces[3, env.feet_indices[0], :2] = torch.tensor([6.0, 8.0])
        env.contact_forces[4, env.feet_indices[1], 0] = 9.0
        expected = baseline * torch.tensor([0, 0, 0, 1, 1, 1])
        for mode in ("on", "off"):
            env.set_guidance_mode(mode)
            torch.testing.assert_close(env._reward_leg_symmetry(), expected)

        # Restores immediately after force disappears, even during a latched swing.
        env.guidance.active_leg[:] = 0
        env.contact_forces[:, env.feet_indices, :2] = 0.0
        torch.testing.assert_close(env._reward_leg_symmetry(), baseline)
        env.dof_pos[:, env.leg_dof_indices] = env.default_dof_pos[:, env.leg_dof_indices]
        torch.testing.assert_close(env._reward_leg_symmetry(), torch.zeros(6))

    def test_dimensions_and_teacher_state_are_absent_from_actor(self):
        env = ctbc_env()
        env.step(torch.zeros(2, 6))
        self.assertEqual((env.obs_buf.shape[1], env.privileged_obs_buf.shape[1],
                          env.obs_history.shape[1]), (31, 151, 155))
        before = env.compute_proprioception_observations().clone()
        critic = env.privileged_obs_buf.clone()
        env.contact_forces[:, env.feet_indices, :2] = 100.0
        env.wheel_force[:] = 123.0
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = 0.3
        env.guidance.start_height[:] = 10.0
        env.ctbc_reference[:] = 0.5
        torch.testing.assert_close(env.compute_proprioception_observations(), before)
        env.compute_observations()
        torch.testing.assert_close(env.obs_buf, before)
        self.assertFalse(torch.equal(env.privileged_obs_buf, critic))

    def test_policy_sample_is_not_modified_by_guidance_or_clipping(self):
        env = ctbc_env()
        env.guidance.active_leg[:] = 0
        env.guidance.elapsed[:] = 0.29
        actions = torch.tensor([[2.5, -2.5, 0.3, 0.1, -0.2, 0.4]]).repeat(2, 1)
        before = actions.clone()
        env.step(actions)
        torch.testing.assert_close(actions, before)
        self.assertLessEqual(env.actions.abs().max().item(), 1.0)

    def test_individual_joint_scales_and_observed_executed_wheel_targets(self):
        env = ctbc_env(permutation=[5, 2, 4, 0, 3, 1])
        env.commands[:, 1] = 0.1
        actions = torch.tensor([[0.4, -0.4, 0.3, -0.2, 0.2, -0.2]]).repeat(2, 1)
        env._compute_torques(actions)
        scales = torch.tensor([0.5, 0.9, 0.5, 0.9])
        expected = env.default_dof_pos[:, env.leg_dof_indices] + actions[:, [0, 1, 3, 4]]*scales
        torch.testing.assert_close(env.position_target[:, env.leg_dof_indices], expected)
        wheel = (actions[:, [2, 5]]*10.0
                 + (0.2-0.1*env.wheel_lateral_offsets)/env.cfg.asset.wheel_radius)
        torch.testing.assert_close(env.velocity_target[:, env.wheel_dof_indices], wheel)
        expected_actions = actions.clone()
        expected_actions[:, [2, 5]] = wheel/10.0
        torch.testing.assert_close(env.ctbc_executed_actions, expected_actions)
        env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:, 21:27], expected_actions)
        torch.testing.assert_close(env.obs_history[:, -31+21:-31+27], expected_actions)

    def test_pure_policy_actions_are_independent_of_collision_force(self):
        first, second = ctbc_env(), ctbc_env()
        first.set_guidance_mode("off")
        second.set_guidance_mode("off")
        first.contact_forces[:, first.feet_indices[0], 0] = 100.0
        second.contact_forces[:, second.feet_indices[1], 0] = 200.0
        actions = torch.full((2, 6), 0.1)
        for _ in range(8):
            first.step(actions)
            second.step(actions)
        self.assertEqual(first.ctbc_feedforward_weight, 0.0)
        torch.testing.assert_close(first.actions, second.actions)
        torch.testing.assert_close(first.position_target, second.position_target)
        torch.testing.assert_close(first.velocity_target, second.velocity_target)
        self.assertFalse(first.guidance.swing_mask.any())
        self.assertFalse(second.guidance.swing_mask.any())

    def test_guidance_changes_only_the_selected_leg_instruction(self):
        guided, baseline = ctbc_env(), ctbc_env()
        guided.set_guidance_mode("on")
        baseline.set_guidance_mode("off")
        for env in (guided, baseline):
            env.contact_forces[:, env.feet_indices[0], 0] = 20.0
        for _ in range(33):
            guided.step(torch.zeros(2, 6))
            baseline.step(torch.zeros(2, 6))
        self.assertTrue((guided.actions[:, 0] > 0.5).all())
        self.assertTrue((guided.actions[:, 1] < -0.6).all())
        torch.testing.assert_close(guided.actions[:, [2, 3, 4, 5]],
                                   baseline.actions[:, [2, 3, 4, 5]])
        self.assertEqual(guided.guidance.active_leg.tolist(), [0, 0])
        self.assertTrue((guided.ctbc_trigger_count == 1).all())

    def test_action_history_and_rate_follow_constant_actual_targets(self):
        env = ctbc_env()
        env.set_guidance_mode("off")
        values = []
        actual_rate = env._reward_action_rate

        def capture_rate():
            value = actual_rate()
            values.append(value.clone())
            return value

        env.reward_functions[env.reward_names.index("action_rate")] = capture_rate
        actions = torch.full((2, 6), 0.1)
        for _ in range(2):
            env.step(actions)
            torch.testing.assert_close(env.last_actions[:, :, 0], env.ctbc_executed_actions)
        self.assertTrue((values[0] > 0).all())
        torch.testing.assert_close(values[1], torch.zeros(2))
        self.assertFalse(torch.equal(env.last_actions[:, [2, 5], 0], actions[:, [2, 5]]))
        torch.testing.assert_close(env.last_actions[:, :, 1], env.ctbc_executed_actions)

    def test_delayed_history_records_applied_target_instead_of_latest_sample(self):
        env = ctbc_env(max_delay_ms=10)
        env.set_guidance_mode("off")
        env.action_delay_idx[:] = 2  # Two 5 ms substeps, one policy frame.
        first = torch.full((2, 6), 0.4)
        env.step(first)
        torch.testing.assert_close(env.ctbc_executed_actions[:, [0, 1, 3, 4]],
                                   torch.zeros(2, 4))
        torch.testing.assert_close(env.last_actions[:, :, 0], env.ctbc_executed_actions)
        env.step(torch.zeros(2, 6))
        torch.testing.assert_close(env.ctbc_executed_actions[:, [0, 1, 3, 4]],
                                   first[:, [0, 1, 3, 4]])
        torch.testing.assert_close(env.last_actions[:, :, 0], env.ctbc_executed_actions)
        torch.testing.assert_close(env.obs_buf[:, 21:27], env.ctbc_executed_actions)

    def test_guidance_and_reset_work_across_separate_inference_frames(self):
        env = ctbc_env()
        env.contact_forces[:, env.feet_indices[0], 0] = 20.0
        for index in range(8):
            # A fresh inference scope per frame catches persistent buffers
            # accidentally replaced by tensors created in an earlier scope.
            with torch.inference_mode():
                actions = torch.zeros(2, 6)
                if index == 6:
                    env.root_states[0, 2] = 0.1  # Exercise a partial reset too.
                obs, critic, _, _, _, history = env.step(actions)
                self.assertTrue(torch.isfinite(obs).all())
                self.assertTrue(torch.isfinite(critic).all())
                self.assertTrue(torch.isfinite(history).all())

    def test_success_requires_both_wheels_goal_height_support_and_dwell(self):
        env = ctbc_env(4)
        place_at_goal(env)
        env.rigid_body_states[1, env.feet_indices[1], 0] = 0.89
        env.rigid_body_states[2, env.feet_indices[1], 2] += 0.05
        env.contact_forces[3, env.feet_indices[1], 2] = 0.0
        for _ in range(14):
            env._post_physics_step_callback()
        self.assertFalse(env.ctbc_success.any())
        env._post_physics_step_callback()
        self.assertEqual(env.ctbc_success.tolist(), [True, False, False, False])
        # A brief support loss resets dwell rather than accumulating intervals.
        env.contact_forces[0, env.feet_indices[1], 2] = 0.0
        env._post_physics_step_callback()
        self.assertEqual(env.ctbc_goal_dwell[0].item(), 0.0)
        self.assertFalse(env.ctbc_success[0])

    def test_physical_failure_and_tile_escape_override_reached_goal(self):
        env = ctbc_env(3)
        place_at_goal(env)
        env.ctbc_success[:] = True
        env.contact_forces[0, 0, 2] = 10.0
        env.root_states[1, 1] = env.cfg.terrain.terrain_width
        env.episode_length_buf[:] = env.max_episode_length
        env.check_termination()
        self.assertEqual(env.ctbc_success.tolist(), [False, False, True])
        self.assertTrue(env.reset_buf.all())
        self.assertFalse(env.time_out_buf.any())
        self.assertTrue(env.physical_failure[0])
        self.assertTrue(env.ctbc_out_of_bounds[1])

    def test_success_reward_is_full_five_once_and_episode_event_survives_reset(self):
        env = ctbc_env(1)
        env.set_guidance_mode("off")
        place_at_goal(env)
        env.ctbc_goal_dwell[:] = 0.14
        # Isolate the completion event from unrelated locomotion reward rates.
        env.reward_scales = {"ctbc_success": 5.0*env.dt}
        env.reward_names = ["ctbc_success"]
        env.reward_functions = [env._reward_ctbc_success]
        env.episode_sums = {"ctbc_success": torch.zeros(1)}
        _, _, reward, done, info, _ = env.step(torch.zeros(1, 6))
        torch.testing.assert_close(reward, torch.tensor([5.0]))
        self.assertTrue(done.item())
        self.assertTrue(info["ctbc_episode"]["success"].item())
        self.assertAlmostEqual(info["ctbc_episode"]["height_m"].item(), 0.05)
        self.assertFalse(env.ctbc_success.item())
        _, _, reward, _, info, _ = env.step(torch.zeros(1, 6))
        torch.testing.assert_close(reward, torch.zeros(1))
        self.assertNotIn("ctbc_episode", info)

    def test_reset_captures_counts_before_clear_and_isolates_other_environment(self):
        env = ctbc_env()
        env.metric_steps[:] = 12
        env.ctbc_success[:] = True
        env.ctbc_trigger_count[:] = torch.tensor([4.0, 7.0])
        env.ctbc_swing_timeouts[:] = torch.tensor([2.0, 3.0])
        env.guidance.active_leg[:] = torch.tensor([0, 1])
        env.guidance.elapsed[:] = 0.2
        env.ctbc_riser_heights[:] = 0.07
        env.reset_idx(torch.tensor([0]))
        event = env.extras["ctbc_episode"]
        self.assertEqual(event["env_ids"].tolist(), [0])
        self.assertTrue(event["success"].item())
        self.assertAlmostEqual(event["height_m"].item(), 0.07)
        self.assertEqual(env.extras["episode"]["ctbc_triggers"].item(), 4.0)
        self.assertEqual(env.extras["episode"]["ctbc_swing_timeouts"].item(), 2.0)
        self.assertEqual(env.ctbc_trigger_count.tolist(), [0.0, 7.0])
        self.assertEqual(env.ctbc_success.tolist(), [False, True])
        self.assertEqual(env.guidance.active_leg.tolist(), [-1, 1])
        self.assertFalse(env.obs_history[0].reshape(5, 31)[:, 21:27].any())

    def test_curriculum_uses_five_completed_attempts_and_rate_thresholds(self):
        env = ctbc_env(3)
        env.custom_origins = True
        env.terrain_levels = torch.ones(3, dtype=torch.long)
        env.terrain_types = torch.zeros(3, dtype=torch.long)
        env.max_terrain_level = 6
        env.terrain_origins = torch.zeros(6, 1, 3)
        env.terrain_origins[:, 0, 0] = torch.arange(6)*6.0
        env.terrain_class_map = torch.full((6, 1), 3, dtype=torch.long)
        env.ctbc_terrain_maps = dict(
            riser_heights=torch.full((6, 1), 0.05),
            tread_widths=torch.full((6, 1), 0.5),
            step_counts=torch.ones(6, 1, dtype=torch.long),
            first_riser_x=env.terrain_origins[:, :, 0]+0.7,
            goal_x=env.terrain_origins[:, :, 0]+1.0)
        env._refresh_climb_targets()
        env.metric_steps[:] = 10
        outcomes = [[True, True, True], [True, False, True],
                    [True, False, False], [True, False, False],
                    [False, False, False]]
        for outcome in outcomes[:4]:
            env.ctbc_success[:] = torch.tensor(outcome)
            env._update_terrain_curriculum(torch.arange(3))
        self.assertEqual(env.terrain_levels.tolist(), [1, 1, 1])
        env.ctbc_success[:] = torch.tensor(outcomes[4])
        env._update_terrain_curriculum(torch.arange(3))
        self.assertEqual(env.terrain_levels.tolist(), [2, 0, 1])
        self.assertEqual(env.success_ids.tolist(), [0])
        self.assertEqual(env.fail_ids.tolist(), [1])
        self.assertFalse(env.ctbc_attempts.any())
        self.assertFalse(env.ctbc_wins.any())
        torch.testing.assert_close(env.env_origins[:, 0], torch.tensor([12.0, 0.0, 6.0]))

    def test_cumulative_iteration_schedule_and_explicit_guidance_modes(self):
        env = ctbc_env()
        env.set_training_iteration(2999)
        self.assertEqual(env.ctbc_feedforward_weight, 1.0)
        env.set_training_iteration(9000)
        self.assertEqual(env.ctbc_feedforward_weight, 0.5)
        env.set_training_iteration(15000)
        self.assertEqual(env.ctbc_feedforward_weight, 0.0)
        env.set_guidance_mode("on")
        self.assertEqual(env.ctbc_feedforward_weight, 1.0)
        env.set_guidance_mode("off")
        self.assertEqual(env.ctbc_feedforward_weight, 0.0)
        with self.assertRaises(ValueError):
            env.set_guidance_mode("invalid")

    def test_real_ppo_keeps_policy_samples_and_restores_cumulative_guidance_schedule(self):
        namespace = dict(NAMESPACE, optim=optim, os=os, time=time,
                         statistics=statistics, deque=deque, VecEnv=object,
                         SummaryWriter=lambda **kw: SimpleNamespace(close=lambda: None))
        for path in ("modules/actor_critic.py", "modules/actor_critic_sequence.py",
                     "storage/rollout_storage.py", "algorithms/ppo.py",
                     "runners/on_policy_runner.py"):
            load_code("wheel_legged_gym/rsl_rl/" + path, namespace)
        config = NS["class_to_dict"](NAMESPACE["Q1CTBCCfgPPO"]())
        config["runner"].update(num_steps_per_env=2, save_interval=1,
                               logger="tensorboard", periodic_video={"enabled": False})
        config["policy"].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8],
                               encoder_hidden_dims=[16, 8], init_noise_std=0.1)
        config["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)

        def make_runner(folder):
            env = ctbc_env(8)
            env.cfg.ctbc.feedforward_hold_iterations = 1
            env.cfg.ctbc.feedforward_anneal_iterations = 2
            # Begin at the useful part of an already latched lift instruction,
            # so even a two-step rollout actually executes teacher assistance.
            env.guidance.active_leg[:] = 0
            env.guidance.elapsed[:] = 0.29
            env.get_observations = lambda: (env.obs_buf, env.obs_history)
            env.get_privileged_observations = lambda: env.privileged_obs_buf
            env.reset = lambda: (env.obs_buf, env.privileged_obs_buf)
            calls = []
            set_iteration = env.set_training_iteration

            def record_iteration(iteration):
                set_iteration(iteration)
                calls.append((iteration, env.ctbc_feedforward_weight))

            env.set_training_iteration = record_iteration
            runner = namespace["OnPolicyRunner"](env, config, folder, device="cpu")
            runner.log = lambda values: None
            transitions, expected = [], {}
            step = env.step
            add_transition = runner.alg.storage.add_transitions

            def checked_step(actions):
                expected["sample"] = actions.clone()
                expected["log_prob"] = runner.alg.transition.actions_log_prob.clone()
                distribution = torch.distributions.Normal(
                    runner.alg.transition.action_mean, runner.alg.transition.action_sigma)
                torch.testing.assert_close(expected["log_prob"],
                                           distribution.log_prob(actions).sum(-1))
                result = step(actions)
                torch.testing.assert_close(actions, expected["sample"])
                expected["guided"] = not torch.equal(
                    env.actions[:, [0, 1, 3, 4]],
                    expected["sample"].clamp(-1, 1)[:, [0, 1, 3, 4]])
                return result

            def checked_transition(transition):
                # This is the real PPO storage insertion, after env.step has
                # applied feedforward. Both sample and log probability must
                # still describe the network distribution, not its teacher.
                torch.testing.assert_close(transition.actions, expected["sample"])
                torch.testing.assert_close(transition.actions_log_prob, expected["log_prob"])
                transitions.append(expected["guided"])
                add_transition(transition)

            env.step = checked_step
            runner.alg.storage.add_transitions = checked_transition
            return runner, calls, transitions

        torch.manual_seed(41)
        with tempfile.TemporaryDirectory(dir=os.environ.get("Q1_TEST_TMP")) as folder:
            first, first_calls, first_transitions = make_runner(folder)
            initial = {name: value.clone() for name, value in
                       first.alg.actor_critic.actor.state_dict().items()}
            first.learn(2)
            self.assertEqual(first_calls, [(0, 1.0), (1, 1.0)])
            self.assertEqual(len(first_transitions), 4)
            self.assertTrue(all(first_transitions))
            self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in
                                first.alg.actor_critic.actor.state_dict().items()))
            checkpoint = Path(folder) / "model_2.pt"
            saved = torch.load(checkpoint)
            self.assertEqual(saved["iter"], 2)
            self.assertTrue(saved["optimizer_state_dict"]["state"])
            self.assertTrue(saved["extra_optimizer_state_dict"]["state"])

            second, second_calls, second_transitions = make_runner(folder)
            second.load(str(checkpoint))
            # Loading must restore guidance immediately, before any rollout.
            self.assertEqual(second_calls, [(2, 0.5)])
            self.assertEqual(second.env.ctbc_feedforward_weight, 0.5)
            for name, value in second.alg.actor_critic.state_dict().items():
                torch.testing.assert_close(value, saved["model_state_dict"][name])
            self.assertTrue(second.alg.extra_optimizer.state_dict()["state"])
            second.learn(1)
            self.assertEqual(second_calls, [(2, 0.5), (2, 0.5)])
            self.assertEqual(len(second_transitions), 2)
            self.assertTrue(all(second_transitions))
            resumed = torch.load(Path(folder) / "model_3.pt")
            self.assertEqual(resumed["iter"], 3)
            previous_steps = [state["step"].item() for state in saved["optimizer_state_dict"]["state"].values()]
            resumed_steps = [state["step"].item() for state in resumed["optimizer_state_dict"]["state"].values()]
            self.assertGreater(min(resumed_steps), max(previous_steps))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
