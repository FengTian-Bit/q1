"""Training command coverage must preserve external evaluation commands."""
import unittest
import torch
from test_q1_tcn import SCOPE, tensor_env


class SequenceTests(unittest.TestCase):
    def test_full_range_amplitudes(self):
        env=tensor_env(22)
        env.cfg.commands.sequence_fraction=1.
        env.cfg.commands.sequence_min_fraction=.15
        env.cfg.commands.ranges.lin_vel_x=[-.6,.6]
        env.episode_length_buf[:]=round(2.5/env.dt)
        env._apply_sequence_commands()
        amplitudes=env.commands[:,0].abs()
        self.assertAlmostEqual(float(amplitudes.min()),.09,places=6)
        self.assertAlmostEqual(float(amplitudes.max()),.6,places=6)
        torch.testing.assert_close(amplitudes[:11],amplitudes[11:])
        torch.testing.assert_close(env.commands[:11,0],-env.commands[11:,0])

    def test_phase_boundaries_and_mirrored_motion(self):
        time=torch.tensor([0.,1.99,2.,4.99,5.,6.99,7.,9.99,10.,12.])
        expected=torch.tensor([0.,0.,.6,.6,0.,0.,-.6,-.6,0.,0.])
        function=SCOPE['flat_sequence_velocity']
        torch.testing.assert_close(function(time,.6,1.),expected)
        torch.testing.assert_close(function(time,.6,-1.),-expected)
        torch.testing.assert_close(function(time+12.,.6,1.),expected)

    def test_selected_envs_reset_and_external_commands(self):
        env=tensor_env(4)
        env.cfg.commands.sequence_fraction=.75
        env.cfg.commands.ranges.lin_vel_x=[-.6,.6]
        env.episode_length_buf[:]=round(2.5/env.dt)
        env.commands[:,:2]=.123
        env._apply_sequence_commands()
        torch.testing.assert_close(env.commands[:3,0],torch.tensor([.51,-.519,.528]))
        torch.testing.assert_close(env.commands[:3,1],torch.zeros(3))
        torch.testing.assert_close(env.commands[3,:2],torch.full((2,),.123))
        env.episode_length_buf.zero_();env._apply_sequence_commands()
        torch.testing.assert_close(env.commands[:3,0],torch.zeros(3))
        env.external_commands=True;env.commands[:,:2]=.234
        env.episode_length_buf[:]=round(8./env.dt)
        env._apply_sequence_commands()
        torch.testing.assert_close(env.commands[:,:2],torch.full((4,2),.234))


if __name__=='__main__':unittest.main()
