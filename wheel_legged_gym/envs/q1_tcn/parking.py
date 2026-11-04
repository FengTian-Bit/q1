"""Parking reference state; simulator position is used only by rewards/metrics."""

import torch


class ParkingState:
    def __init__(self, num_envs, device):
        self.reference = torch.zeros(num_envs, 2, device=device)
        self.brake_start = torch.zeros_like(self.reference)
        self.stationary = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_stand = torch.ones_like(self.stationary)
        self.locked = torch.zeros_like(self.stationary)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.weight = torch.zeros_like(self.elapsed)
        self.error = torch.zeros_like(self.elapsed)
        self.braking_distance = torch.zeros_like(self.elapsed)

    def reset(self, ids, position):
        self.reference[ids] = position[ids, :2]
        self.brake_start[ids] = position[ids, :2]
        self.stationary[ids] = False
        self.initial_stand[ids] = True
        self.locked[ids] = False
        self.elapsed[ids] = 0
        self.weight[ids] = 0
        self.error[ids] = 0
        self.braking_distance[ids] = 0

    def update(self, position, commands, dt, cfg):
        stationary = ((commands[:, 0].abs() < cfg.parking_speed_threshold)
                      & (commands[:, 1].abs() < cfg.parking_yaw_threshold))
        entering = stationary & ~self.stationary
        self.elapsed[entering] = 0
        self.brake_start[entering] = position[entering, :2]
        self.braking_distance[entering] = 0
        self.locked[entering] = self.initial_stand[entering]
        self.locked[~stationary] = False
        self.initial_stand[~stationary] = False
        # Entry frame is time zero; lock at a fixed time, never conditional on speed.
        self.elapsed = torch.where(stationary, self.elapsed + (~entering).float()*dt,
                                   torch.zeros_like(self.elapsed))
        lock = stationary & ~self.locked & (self.elapsed >= cfg.parking_brake_s-1e-6)
        self.reference[lock] = position[lock, :2]
        self.braking_distance[lock] = (position[lock, :2]-self.brake_start[lock]).norm(dim=-1)
        self.locked[lock] = True
        self.weight = ((self.elapsed-cfg.parking_brake_s)/cfg.parking_ramp_s).clamp(0, 1)
        self.weight *= (stationary & self.locked).float()
        self.error = (position[:, :2]-self.reference).norm(dim=-1)
        self.error *= (stationary & self.locked).float()
        self.stationary[:] = stationary

    def penalty(self, cfg):
        error = (self.error-cfg.parking_deadband_m).clamp_min(0)/cfg.parking_scale_m
        return self.weight*error.square().clamp(max=1)
