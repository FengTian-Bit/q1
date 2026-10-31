"""Record applied Q1 joint torques and save a PNG without opening a GUI."""

from array import array
from pathlib import Path


def _values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class TorqueRecorder:
    """Keep one robot's six applied torques at every physics step, in N m."""

    def __init__(self, env, env_index=0):
        if not 0 <= env_index < env.num_envs:
            raise ValueError("Torque recorder environment index is out of range")
        self.env_index = env_index
        self.dt = float(env.cfg.sim.dt)
        self.indices = [int(index) for index in _values(env.policy_dof_indices)]
        self.joint_names = [env.dof_names[index] for index in self.indices]
        self.limits = _values(env.torque_limits[env.policy_dof_indices])
        if len(self.indices) != 6:
            raise ValueError("Q1 torque plots require six policy joints")
        # A flat numeric buffer avoids keeping GPU tensors or a Python object
        # for every sample during long playback sessions.
        self.samples = array("d")

    @property
    def count(self):
        return len(self.samples) // 6

    def record(self, torques):
        self.samples.extend(_values(torques[self.env_index, self.indices]))

    def save_png(self, path, title="Q1 applied joint torques"):
        if not self.count:
            print("No torque samples recorded; skipping torque PNG.", flush=True)
            return None
        import numpy as np
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torques = np.frombuffer(self.samples, dtype=np.float64).reshape(-1, 6)
        times = (np.arange(self.count) + 1) * self.dt
        figure = Figure(figsize=(14, 9), facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 2, sharex=True)
        labels = ("Left thigh", "Left calf", "Left wheel", "Right thigh", "Right calf", "Right wheel")
        for index, (label, name, limit) in enumerate(zip(labels, self.joint_names, self.limits)):
            axis = axes[index % 3, index // 3]
            axis.plot(times, torques[:, index], color="#2563eb" if index < 3 else "#d97706",
                      linewidth=0.8, label="Applied torque")
            axis.axhline(limit, color="#dc2626", linestyle="--", linewidth=0.8, label="Torque limits")
            axis.axhline(-limit, color="#dc2626", linestyle="--", linewidth=0.8)
            axis.axhline(0, color="#64748b", linewidth=0.5)
            axis.set_title("{} ({})".format(label, name), fontsize=10)
            axis.set_ylabel("Torque [N m]")
            axis.grid(alpha=0.2)
            axis.margins(x=0.01)
            if index % 3 == 2:
                axis.set_xlabel("Simulation time [s]")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        figure.suptitle(title, fontsize=14, y=0.98)
        figure.text(0.5, 0.025, "Environment {} | Physics sampling: {:.4f} s | {} samples | Applied actuator torque, not contact torque".format(
            self.env_index, self.dt, self.count), ha="center", fontsize=9, color="#475569")
        figure.tight_layout(rect=(0, 0.045, 1, 0.95))
        try:
            with rc_context({"agg.path.chunksize": 10000}):
                figure.savefig(str(path), dpi=160, format="png")
        finally:
            figure.clear()
        print("Q1 torque PNG:", path, flush=True)
        return path
