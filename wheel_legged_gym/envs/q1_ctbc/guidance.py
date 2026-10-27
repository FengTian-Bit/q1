"""Contact-driven swing instructions, independent of the Isaac Gym runtime.

The caller supplies one sample per policy step. All state is per environment,
so resets and asynchronous obstacle encounters do not couple parallel robots.
Joint offsets use policy leg order: left hip, left knee, right hip, right knee.
"""

import math

import torch


class ContactGuidance:
    def __init__(self, num_envs, device, dt, cfg):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.cfg = cfg
        self.contact_threshold = float(getattr(cfg, "contact_threshold", 12.0))
        self.contact_frames = int(getattr(cfg, "contact_frames", 3))
        self.swing_duration = float(getattr(cfg, "swing_duration", 0.6))
        self.min_swing_time = float(getattr(cfg, "min_swing_time", 0.3))
        self.min_landing_rise = float(getattr(cfg, "min_landing_rise", 0.0))
        self.cooldown_s = float(getattr(cfg, "cooldown_s", 0.2))
        ramp = float(getattr(cfg, "feedforward_ramp_iterations", 0))
        hold = float(getattr(cfg, "feedforward_hold_iterations", 3000))
        if not math.isfinite(ramp) or not 0 <= ramp <= hold:
            raise ValueError("Feedforward ramp must be between zero and the hold duration")
        if (self.num_envs < 1 or self.dt <= 0 or self.contact_frames < 1
                or self.contact_threshold < 0 or self.swing_duration <= 0
                or not 0 <= self.min_swing_time <= self.swing_duration
                or self.cooldown_s < 0 or not math.isfinite(self.min_landing_rise)
                or self.min_landing_rise < 0):
            raise ValueError("Invalid contact-guidance timing or threshold configuration")

        self.active_leg = torch.full((self.num_envs,), -1, device=self.device,
                                     dtype=torch.long)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.cooldown = torch.zeros_like(self.elapsed)
        self.contact_count = torch.zeros(self.num_envs, 2, device=self.device,
                                         dtype=torch.long)
        self.has_unloaded = torch.zeros(self.num_envs, device=self.device,
                                         dtype=torch.bool)
        self.triggered = torch.zeros(self.num_envs, 2, device=self.device,
                                      dtype=torch.bool)
        self.timed_out = torch.zeros_like(self.has_unloaded)
        self.start_height = torch.zeros(self.num_envs, 2, device=self.device)
        self.amplitudes = torch.tensor([
            getattr(cfg, "hip_amplitude", 0.2952),
            getattr(cfg, "knee_amplitude", -0.5905),
        ], device=self.device, dtype=self.elapsed.dtype).repeat(self.num_envs, 1)
        if not torch.isfinite(self.amplitudes).all():
            raise ValueError("Contact-guidance amplitudes must be finite")
        self._leg_ids = torch.arange(2, device=self.device)

    def set_amplitudes(self, env_ids, amplitudes):
        """Set per-environment hip/knee offsets without coupling other robots."""
        values = torch.as_tensor(amplitudes, device=self.device,
                                 dtype=self.amplitudes.dtype).detach()
        expected = self.amplitudes[env_ids].shape
        if values.ndim != 2 or values.shape != expected:
            raise ValueError("Contact-guidance amplitudes must have shape [selected_envs, 2]")
        if not torch.isfinite(values).all():
            raise ValueError("Contact-guidance amplitudes must be finite")
        self.amplitudes[env_ids] = values

    @property
    def swing_mask(self):
        return self.active_leg[:, None] == self._leg_ids[None, :]

    @property
    def phase(self):
        return (self.elapsed / self.swing_duration).clamp(0.0, 1.0)

    def update(self, force_xy, support, foot_bottom_z, moving):
        """Advance existing swings, then debounce new horizontal collisions.

        A swing remains latched when collision force disappears. Successful
        landing requires an observed unloading followed by support after the
        minimum swing time; continued initial support cannot finish a swing.
        Reaching the duration ends even an unsuccessful swing and starts the
        cooldown. ``triggered`` and ``timed_out`` are one-step event pulses.
        """
        expected = (self.num_envs, 2)
        if (tuple(force_xy.shape) != expected or tuple(support.shape) != expected
                or tuple(foot_bottom_z.shape) != expected
                or tuple(moving.shape) != (self.num_envs,)):
            raise ValueError("Contact samples must be [num_envs, 2], moving [num_envs]")
        force_xy = force_xy.to(device=self.device)
        support = support.to(device=self.device, dtype=torch.bool)
        foot_bottom_z = foot_bottom_z.to(device=self.device)
        moving = moving.to(device=self.device, dtype=torch.bool)
        self.triggered.zero_()
        self.timed_out.zero_()
        self.cooldown.sub_(self.dt).clamp_(min=0.0)

        was_active = self.active_leg >= 0
        self.elapsed[was_active] += self.dt
        active_support = support.gather(1, self.active_leg.clamp_min(0)[:, None]).squeeze(1)
        self.has_unloaded |= was_active & ~active_support
        # A small tolerance prevents float32 accumulation adding an extra tick.
        tolerance = 1e-6 * max(self.swing_duration, 1.0)
        active_leg = self.active_leg.clamp_min(0)[:, None]
        landing_height = foot_bottom_z.gather(1, active_leg).squeeze(1)
        initial_height = self.start_height.gather(1, active_leg).squeeze(1)
        climbed = landing_height >= initial_height + self.min_landing_rise - tolerance
        landed = (was_active & self.has_unloaded & active_support
                  & (self.elapsed >= self.min_swing_time - tolerance) & climbed)
        timeout = was_active & (self.elapsed >= self.swing_duration - tolerance)
        finished = landed | timeout
        self.timed_out[:] = timeout & ~landed
        self.active_leg[finished] = -1
        self.elapsed[finished] = 0.0
        self.cooldown[finished] = self.cooldown_s
        self.has_unloaded[finished] = False

        eligible = ~was_active & (self.cooldown <= 0.0) & moving
        if not getattr(self.cfg, "trigger_enabled", True):
            eligible[:] = False
        contact = force_xy > self.contact_threshold
        self.contact_count[:] = torch.where(
            contact & eligible[:, None], self.contact_count + 1,
            torch.zeros_like(self.contact_count)).clamp_max(self.contact_frames)
        stable = self.contact_count >= self.contact_frames
        begin = eligible & stable.any(dim=1)
        # Only stable candidates participate: one transient larger impact must
        # not steal a trigger from the other wheel's sustained collision.
        candidate_force = force_xy.masked_fill(~stable, -torch.inf)
        selected = candidate_force.argmax(dim=1)
        rows = begin.nonzero(as_tuple=False).flatten()
        self.active_leg[rows] = selected[rows]
        self.elapsed[rows] = 0.0
        self.has_unloaded[rows] = False
        self.start_height[rows] = foot_bottom_z[rows]
        self.triggered[rows, selected[rows]] = True
        self.contact_count[rows] = 0

    def reference_offsets(self):
        """Return an unweighted cosine instruction in radians, [N, 4]."""
        shape = 0.5 * (1.0 - torch.cos(2.0 * math.pi * self.phase))
        shape = shape[:, None] * self.swing_mask
        return (shape[:, :, None] * self.amplitudes[:, None, :]).reshape(self.num_envs, 4)

    def feedforward_weight(self, iteration):
        """Optionally ramp guidance in, hold it, then linearly remove it.

        The hold duration includes the ramp, and all durations are relative to
        the saved start iteration. A zero ramp preserves the original schedule.
        """
        if not getattr(self.cfg, "feedforward_enabled", True):
            return 0.0
        initial = float(getattr(self.cfg, "feedforward_initial", 1.0))
        start = float(getattr(self.cfg, "feedforward_start_iteration", 0))
        hold = float(getattr(self.cfg, "feedforward_hold_iterations", 3000))
        anneal = float(getattr(self.cfg, "feedforward_anneal_iterations", 12000))
        ramp = float(getattr(self.cfg, "feedforward_ramp_iterations", 0))
        if not math.isfinite(ramp) or not 0 <= ramp <= hold:
            raise ValueError("Feedforward ramp must be between zero and the hold duration")
        training_elapsed = max(float(iteration) - start, 0.0)
        if ramp > 0 and training_elapsed < ramp:
            return initial * training_elapsed / ramp
        elapsed = max(training_elapsed - hold, 0.0)
        if anneal <= 0:
            return initial if training_elapsed < hold else 0.0
        return initial * max(0.0, min(1.0, 1.0 - elapsed / anneal))

    def reset(self, env_ids):
        """Clear selected environments without disturbing ongoing neighbours."""
        self.active_leg[env_ids] = -1
        for field in (self.elapsed, self.cooldown, self.contact_count,
                      self.has_unloaded, self.triggered, self.timed_out,
                      self.start_height):
            field[env_ids] = 0
