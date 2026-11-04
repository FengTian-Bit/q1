"""First-episode metrics: failed robots never re-enter the evaluation cohort."""

import numpy as np


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=float)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


class TrialMetrics:
    def __init__(self, count, dt, initial_position, enabled, push_end=None):
        self.count, self.dt, self.enabled = count, dt, enabled
        self.initial_position = np.array(initial_position, copy=True)
        self.active = np.ones(count, dtype=bool)
        self.failed = np.zeros(count, dtype=bool)
        self.steps = np.zeros(count)
        self.sums = {name: np.zeros(count) for name in
            ("speed_sq", "pitch_sq", "roll_sq", "slip_abs", "support_count", "saturation",
             "velocity_est_sq", "velocity_est_count", "slip_est_sq", "slip_est_count",
             "tp", "fp", "fn", "tn", "high_slip", "abnormal_spin",
             "contact_delay", "contact_detected", "contact_events")}
        self.max_pitch = np.zeros(count)
        self.max_roll = np.zeros(count)
        self.max_drift = np.zeros(count)
        self.end_drift = np.zeros(count)
        self.push_end = push_end
        self.recovery_time = np.full(count, np.nan)
        self.stable_steps = np.zeros(count, dtype=int)
        self.exposed = np.zeros(count, dtype=bool)
        self.previous_contact = np.zeros((count, 2), dtype=bool)
        self.previous_contact_valid = np.zeros((count, 2), dtype=bool)
        self.pending_contact = np.zeros((count, 2), dtype=bool)
        self.contact_event_step = np.zeros((count, 2))
        self.previous_command = np.zeros(count)
        self.command_event_time = np.zeros(count)
        self.command_pending = np.zeros(count, dtype=bool)
        self.command_events = np.zeros(count)
        self.command_settled = np.zeros(count)
        self.command_stable = np.zeros(count)
        self.command_settling_sum = np.zeros(count)
        self.trace = []
        self.parking_steps = np.zeros(count)
        self.parking_sq = np.zeros(count)
        self.parking_max = np.zeros(count)
        self.parking_end = np.zeros(count)
        self.braking_max = np.zeros(count)
        self.clipping_sum = np.zeros(count)
        self.zero_steps = np.zeros(count)
        self.zero_speed_sq = np.zeros(count)

    def update_estimates(self, estimates, targets):
        valid = self.active & (targets[:, 7] > 0.5)
        if self.enabled[0]:
            error = ((estimates["velocity"]-targets[:, :3])**2).sum(-1)
            self.sums["velocity_est_sq"] += error*valid
            self.sums["velocity_est_count"] += 3*valid
        if self.enabled[1]:
            predicted = estimates["contact_probability"] >= 0.5
            truth = targets[:, 3:5] > 0.5
            mask = valid[:, None]
            changed = mask & self.previous_contact_valid & (truth != self.previous_contact)
            self.sums["contact_events"] += changed.sum(-1)
            self.contact_event_step[changed] = np.broadcast_to(self.steps[:, None], changed.shape)[changed]
            self.pending_contact |= changed
            detected = self.pending_contact & mask & (predicted == truth)
            delay = (self.steps[:, None]-self.contact_event_step)*self.dt
            self.sums["contact_delay"] += (delay*detected).sum(-1)
            self.sums["contact_detected"] += detected.sum(-1)
            self.pending_contact[detected] = False
            self.previous_contact[valid] = truth[valid]
            self.previous_contact_valid[valid] = True
            for name, term in (("tp", predicted & truth), ("fp", predicted & ~truth),
                               ("fn", ~predicted & truth), ("tn", ~predicted & ~truth)):
                self.sums[name] += (term & mask).sum(-1)
        if self.enabled[2]:
            mask = valid[:, None] & (targets[:, 8:10] > 0.5)
            self.sums["slip_est_sq"] += (((estimates["slip"]-targets[:, 5:7])**2)*mask).sum(-1)
            self.sums["slip_est_count"] += mask.sum(-1)

    def update(self, time_s, sample):
        active = self.active.copy()
        self.steps += active
        parking_valid = active & sample.get("parking_valid", np.zeros(self.count, dtype=bool))
        parking_error = sample.get("parking_error", np.zeros(self.count))
        self.parking_steps += parking_valid
        self.parking_sq += parking_error**2*parking_valid
        self.parking_max = np.maximum(self.parking_max, parking_error*parking_valid)
        self.parking_end[parking_valid] = parking_error[parking_valid]
        self.braking_max = np.maximum(self.braking_max, sample.get("braking_distance", np.zeros(self.count))*active)
        clipping = sample.get("action_clipping", np.zeros(self.count))
        self.clipping_sum += clipping*active
        zero = active & (np.abs(sample["command"]) < 0.02)
        self.zero_steps += zero
        self.zero_speed_sq += sample["velocity"][:, 0]**2*zero
        error = sample["velocity"][:, 0]-sample["command"]
        changed = active & (np.abs(sample["command"]-self.previous_command) > 1e-6)
        self.command_events += changed
        self.command_event_time[changed] = time_s-self.dt
        self.command_pending[changed] = True
        self.command_stable[changed] = 0
        stable = active & (np.abs(error) < 0.05) & ~sample["failure"]
        self.command_stable = np.where(stable, self.command_stable+1, 0)
        settled = self.command_pending & (self.command_stable*self.dt >= 0.5)
        settled &= active & ~sample["failure"]
        self.command_settled += settled
        self.command_settling_sum += (time_s-self.command_event_time)*settled
        self.command_pending[settled] = False
        self.previous_command[active] = sample["command"][active]
        self.sums["speed_sq"] += error**2*active
        for angle in ("pitch", "roll"):
            self.sums[angle+"_sq"] += sample[angle]**2*active
        support = sample["support"] & active[:, None]
        self.sums["slip_abs"] += (np.abs(sample["slip"])*support).sum(-1)
        self.sums["support_count"] += support.sum(-1)
        high_slip = (np.abs(sample["slip"]) > 0.2) & support
        self.sums["high_slip"] += high_slip.sum(-1)
        self.sums["abnormal_spin"] += (high_slip & (np.abs(sample["velocity"][:, :1]) < 0.1)).any(-1)*self.dt
        self.sums["saturation"] += sample["saturation"]*active
        self.max_pitch = np.maximum(self.max_pitch, np.abs(sample["pitch"])*active)
        self.max_roll = np.maximum(self.max_roll, np.abs(sample["roll"])*active)
        drift = np.linalg.norm(sample["position"][:, :2]-self.initial_position[:, :2], axis=-1)
        self.max_drift = np.maximum(self.max_drift, drift*active)
        self.end_drift[active] = drift[active]
        if self.push_end is not None:
            if self.push_end-0.15 < time_s <= self.push_end:
                self.exposed |= active
            if time_s >= self.push_end:
                stable = active & (np.abs(error) < 0.05) & (np.abs(sample["pitch"]) < 0.12)
                stable &= (np.abs(sample["roll"]) < 0.12) & ~sample["failure"]
                self.stable_steps = np.where(stable, self.stable_steps+1, 0)
                recovered = (self.stable_steps*self.dt >= 0.5) & np.isnan(self.recovery_time)
                recovered &= self.exposed & (time_s-self.push_end <= 3.0)
                self.recovery_time[recovered] = time_s-self.push_end
        live_velocity = sample["velocity"][active, 0]
        mean = lambda array: float(np.mean(array[active])) if active.any() else np.nan
        self.trace.append(dict(time_s=time_s, alive_count=int(active.sum()),
            command_m_s=mean(sample["command"]), velocity_m_s=mean(sample["velocity"][:, 0]),
            velocity_std_m_s=float(live_velocity.std()) if active.any() else np.nan,
            pitch_rad=mean(sample["pitch"]), roll_rad=mean(sample["roll"]),
            slip_abs_m_s=float(np.abs(sample["slip"])[support].mean()) if support.any() else np.nan,
            support_fraction=float(support[active].mean()) if active.any() else np.nan,
            force_n=mean(sample["force"].sum(-1))))
        self.trace[-1]["parking_error_m"] = float(parking_error[parking_valid].mean()) if parking_valid.any() else np.nan
        self.trace[-1]["action_clipping_fraction"] = mean(clipping)
        self.failed |= active & sample["failure"]
        self.active &= ~sample["failure"]

    def rows(self, seed, scenario):
        s = self.sums
        fields = dict(
            observed_s=self.steps*self.dt, failed=self.failed.astype(float),
            speed_rmse_m_s=np.sqrt(divide(s["speed_sq"], self.steps)),
            pitch_rms_rad=np.sqrt(divide(s["pitch_sq"], self.steps)),
            roll_rms_rad=np.sqrt(divide(s["roll_sq"], self.steps)),
            max_pitch_rad=self.max_pitch, max_roll_rad=self.max_roll,
            end_displacement_m=self.end_drift, max_displacement_m=self.max_drift,
            locked_parking_rms_m=np.sqrt(divide(self.parking_sq, self.parking_steps)),
            locked_parking_max_m=np.where(self.parking_steps > 0, self.parking_max, np.nan),
            locked_parking_end_m=np.where(self.parking_steps > 0, self.parking_end, np.nan),
            locked_parking_observed_s=self.parking_steps*self.dt,
            braking_displacement_max_m=self.braking_max,
            zero_command_speed_rmse_m_s=np.sqrt(divide(self.zero_speed_sq, self.zero_steps)),
            action_clipping_fraction=divide(self.clipping_sum, self.steps),
            contact_slip_mae_m_s=divide(s["slip_abs"], s["support_count"]),
            supported_high_slip_fraction=divide(s["high_slip"], s["support_count"]),
            abnormal_spin_s=s["abnormal_spin"],
            wheel_support_fraction=divide(s["support_count"], 2*self.steps),
            actuator_saturation_fraction=divide(s["saturation"], self.steps),
            velocity_est_rmse_m_s=np.sqrt(divide(s["velocity_est_sq"], s["velocity_est_count"])),
            contact_f1=divide(2*s["tp"], 2*s["tp"]+s["fp"]+s["fn"]),
            contact_negative_recall=divide(s["tn"], s["tn"]+s["fp"]),
            contact_positive_fraction=divide(s["tp"]+s["fn"],
                s["tp"]+s["fn"]+s["tn"]+s["fp"]),
            contact_detection_delay_s=divide(s["contact_delay"], s["contact_detected"]),
            contact_transition_detection_fraction=divide(s["contact_detected"], s["contact_events"]),
            command_settling_time_s=divide(self.command_settling_sum, self.command_settled),
            command_settled_fraction=divide(self.command_settled, self.command_events),
            slip_est_rmse_m_s=np.sqrt(divide(s["slip_est_sq"], s["slip_est_count"])),
        )
        if self.push_end is not None:
            # Recoveries followed by another fall do not count as successful trials.
            fields.update(push_exposed=self.exposed.astype(float),
                recovered=(np.isfinite(self.recovery_time) & ~self.failed).astype(float),
                recovery_time_s=np.where(self.failed, np.nan, self.recovery_time))
        return [dict(seed=seed, scenario=scenario, trial=i,
                     **{name: float(value[i]) for name, value in fields.items()})
                for i in range(self.count)]


def summarize(rows):
    """Summarize simulation trials, not independent training-seed uncertainty."""
    result = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        summary = {"trials": len(selected), "metrics": {}}
        keys = sorted(set().union(*(row.keys() for row in selected)) - {"seed", "scenario", "trial"})
        for key in keys:
            values = np.array([row.get(key, np.nan) for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            summary["metrics"][key] = dict(n=len(values),
                mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values) > 1 else None)
        result[scenario] = summary
    return result
