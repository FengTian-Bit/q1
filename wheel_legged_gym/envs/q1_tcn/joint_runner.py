"""Joint PPO lifecycle, isolated from historical frozen-parking experiments."""
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .runner import TCNPpoRunner, config_dict, load_checkpoint
from .joint_task import JointActorCriticTCN
from .ppo import PPO
from .task_config import PROFILE_VERSION


def training_contract(snapshot):
    """Runtime locations, seed and batch count may change on explicit resume."""
    result = deepcopy(snapshot)
    result["env"].pop("seed", None)
    result["env"]["env"].pop("num_envs", None)
    result["train"].pop("seed", None)
    # Runner options change logging/checkpoint cadence, not reward or policy.
    result["train"].pop("runner", None)
    return result


class JointTCNPpoRunner(TCNPpoRunner):
    def __init__(self, env, train_cfg, log_dir, device="cpu", writer=None):
        self.env, self.cfg = env, train_cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.model = JointActorCriticTCN(env.num_obs, env.num_privileged_obs,
            env.num_actions, ablation=train_cfg.ablation,
            **config_dict(train_cfg.policy)).to(self.device)
        self.alg = PPO(self.model, **config_dict(train_cfg.algorithm))
        self.iteration, self.total_steps = 0, 0
        self.snapshot = dict(task="q1_tcn", schema_version=1,
            profile_version=PROFILE_VERSION, env=config_dict(env.cfg), train=config_dict(train_cfg))
        (self.log_dir / "resolved_config.json").write_text(
            json.dumps(self.snapshot, indent=2), encoding="utf-8")

    def load(self, path):
        checkpoint = load_checkpoint(path, self.device)
        saved = checkpoint["config"]
        if saved.get("profile_version") != PROFILE_VERSION:
            raise ValueError("Resume requires a checkpoint from " + PROFILE_VERSION)
        if training_contract(saved) != training_contract(self.snapshot):
            raise ValueError("Joint resume configuration mismatch: reward, control or network changed")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.alg.learning_rate = checkpoint["learning_rate"]
        self.iteration, self.total_steps = checkpoint["iteration"], checkpoint["total_steps"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

    def learn(self, iterations):
        self.model.train()
        with (self.log_dir / "training.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = None
            for _ in range(iterations):
                batch, last_values, failures, timeouts = self.collect()
                metrics = self.alg.update(batch, last_values)
                self.iteration += 1
                self.total_steps += self.env.num_envs*self.cfg.runner.num_steps_per_env
                row = dict(iteration=self.iteration, total_steps=self.total_steps,
                    mean_training_reward=float(batch["environment_rewards"].mean()),
                    failures=failures, timeouts=timeouts,
                    learning_rate=self.alg.learning_rate, **metrics)
                row.update(self.rollout_reward_metrics)
                row.update(self.rollout_acceleration_peaks)
                row["action_clipping_fraction"] = self.rollout_action_clipping
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if stream.tell() == 0:
                        writer.writeheader()
                writer.writerow(row)
                stream.flush()
                if self.writer is not None:
                    for name, value in row.items():
                        self.writer.add_scalar("Train/"+name, value, self.iteration)
                if self.iteration == 1 or self.iteration % 10 == 0:
                    print("q1_tcn joint iter={} reward={:.4f} failures={}".format(
                        self.iteration, row["mean_training_reward"], failures), flush=True)
                if self.iteration % self.cfg.runner.save_interval == 0:
                    self.save()
        print("Saved {}".format(self.save()), flush=True)
