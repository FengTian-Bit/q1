"""Kinematic invariance checks for deployable flat-wheel odometry."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('wheel_odom',ROOT/'wheel_legged_gym/envs/q1_tcn/odometry.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class OdometryTests(unittest.TestCase):
    def inputs(self):
        return torch.tensor([[.6,-1.2,0.,.6,-1.2,0.]]),torch.tensor([[0.,0.,-1.]]),torch.zeros(1,3)

    def test_straight_rolling(self):
        q,g,w=self.inputs();odom=module.WheelOdometry(1,'cpu');odom.reset(torch.tensor([0]),q,g)
        q[:,[2,5]]+=.1
        xy=odom.update(q,w,g,.01)
        torch.testing.assert_close(xy,torch.tensor([[.007496,0.]]),atol=1e-7,rtol=0)

    def test_leg_motion_with_compensating_roll_keeps_base_stationary(self):
        q,g,w=self.inputs();odom=module.WheelOdometry(1,'cpu');odom.reset(torch.tensor([0]),q,g)
        previous=odom.centers(q,odom.rp)
        q[:,[0,3]]+=.02
        current=odom.centers(q,odom.rp)
        q[:,[2,5]]+=(current[:,:,0]-previous[:,:,0])/odom.radius-.02
        xy=odom.update(q,w,g,.01)
        torch.testing.assert_close(xy,torch.zeros_like(xy),atol=1e-7,rtol=0)

    def test_encoder_wrap_does_not_jump(self):
        q,g,w=self.inputs();q[:,[2,5]]=3.13
        odom=module.WheelOdometry(1,'cpu');odom.reset(torch.tensor([0]),q,g)
        q[:,[2,5]]=-3.13
        xy=odom.update(q,w,g,.01)
        self.assertLess(float(xy.norm()),.003)
        self.assertGreater(float(xy[0,0]),0)

    def test_differential_rolling_turns_in_correct_direction(self):
        q,g,w=self.inputs()
        odom=module.WheelOdometry(1,'cpu');odom.reset(torch.tensor([0]),q,g)
        q[:,2]-=.01
        q[:,5]+=.01
        odom.update(q,w,g,.01)
        expected=.02*odom.radius/(.14927+.14973)
        self.assertAlmostEqual(float(odom.yaw[0]),expected,places=6)
        # Q1 base origin is offset from the wheel axle: an in-place wheel turn
        # traces a small base-origin arc, rather than zero base translation.
        center=odom.centers(q,odom.rp).mean(1)[0]
        angle=odom.yaw[0]
        exact=torch.stack((center[0]*(1-angle.cos())+center[1]*angle.sin(),
                           center[1]*(1-angle.cos())-center[0]*angle.sin()))
        torch.testing.assert_close(odom.xy[0],exact,atol=1e-8,rtol=0)

    def test_partial_reset_keeps_other_robot_reference(self):
        q,g,w=self.inputs()
        q,g,w=q.repeat(2,1),g.repeat(2,1),w.repeat(2,1)
        odom=module.WheelOdometry(2,'cpu');odom.reset(torch.arange(2),q,g)
        q[:,[2,5]]+=.1
        odom.update(q,w,g,.01)
        before=odom.xy[1].clone()
        odom.reset(torch.tensor([0]),q,g)
        torch.testing.assert_close(odom.xy[1],before)
        self.assertEqual(float(odom.xy[0].norm()),0)


if __name__=='__main__':unittest.main()
