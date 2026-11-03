"""Test the sequential batch lifecycle without launching training or importing Gym."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("q1_tcn_batch", ROOT / "wheel_legged_gym/envs/q1_tcn/batch.py")
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def checkpoint(path, iteration, seed, ablation, num_envs):
    config = dict(task="q1_tcn", schema_version=1,
        env=dict(seed=seed, env=dict(num_envs=num_envs)),
        train=dict(seed=seed, ablation=ablation, runner=dict(num_steps_per_env=48)))
    info = dict(iteration=iteration, total_steps=iteration*num_envs*48, config=config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test checkpoint placeholder")
    path.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
    (path.parent / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    return info


def read_checkpoint(path):
    return BATCH.read_json(path.with_suffix(".json"))


class FakeProcesses:
    def __init__(self, failure=None):
        self.commands = []
        self.failure = failure

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        self.cwd = kwargs["cwd"]
        if self.failure:
            return SimpleNamespace(returncode=self.failure)
        value = lambda option: command[command.index(option)+1]
        if BATCH.MODULE+"train" in command:
            output = Path(value("--output_dir"))
            previous = read_checkpoint(Path(value("--resume_checkpoint")))["iteration"] if "--resume_checkpoint" in command else 0
            iteration = previous+int(value("--max_iterations"))
            checkpoint(output / "model_{}.pt".format(iteration), iteration,
                int(value("--seed")), value("--ablation"), int(value("--num_envs")))
        elif BATCH.MODULE+"evaluate" in command:
            output = Path(value("--output_dir"))
            output.mkdir(parents=True, exist_ok=True)
            model = Path(value("--checkpoint_path"))
            info = read_checkpoint(model)
            start, end = command.index("--eval_seeds")+1, command.index("--duration")
            summary = dict(checkpoint=str(model), training_seed=info["config"]["train"]["seed"],
                ablation=info["config"]["train"]["ablation"],
                evaluation=dict(seeds=[int(item) for item in command[start:end]],
                    num_envs=int(value("--num_envs")), duration_s=float(value("--duration")), pulse_force_n=8.0),
                scenarios={scenario: {} for scenario in BATCH.SCENARIOS})
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for name in ["summary.csv", "trials.csv"] + [scenario+".png" for scenario in BATCH.SCENARIOS]:
                (output / name).write_bytes(b"test output placeholder")
        return SimpleNamespace(returncode=0)


class BatchTests(unittest.TestCase):
    def test_default_plan_is_15_runs_and_writes_nothing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            output = Path(folder)/"not_created"
            args = BATCH.parse_args(["--dry_run", "--output_root", str(output)])
            stream = io.StringIO()
            with redirect_stdout(stream):
                BATCH.run_batch(args)
            self.assertEqual(stream.getvalue().count(BATCH.MODULE+"train "), 15)
            self.assertEqual(stream.getvalue().count(BATCH.MODULE+"evaluate "), 15)
            self.assertFalse(output.exists())

    def test_all_groups_sequential_evaluation_and_rerun_skip(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1", "--num_envs", "4"])
            process = FakeProcesses()
            with redirect_stdout(io.StringIO()):
                BATCH.run_batch(args, read_checkpoint, process)
            self.assertEqual(len(process.commands), 11)
            for index, ablation in enumerate(BATCH.ABLATIONS):
                command = process.commands[index*2]
                self.assertEqual(command[command.index("--ablation")+1], ablation)
                self.assertIn(BATCH.MODULE+"train", command)
                self.assertIn(BATCH.MODULE+"evaluate", process.commands[index*2+1])
            state = BATCH.read_json(Path(folder)/"ablation_batch.json")
            self.assertEqual(state["status"], "complete")
            self.assertTrue(all(job["status"] == "complete" for job in state["jobs"].values()))
            process.commands.clear()
            with redirect_stdout(io.StringIO()):
                BATCH.run_batch(args, read_checkpoint, process)
            self.assertEqual(len(process.commands), 1)  # comparison only
            self.assertIn("--inputs", process.commands[0])

    def test_resume_uses_remaining_budget_and_preserves_old_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1", "--ablations", "A1",
                                    "--num_envs", "4", "--train_only"])
            previous = Path(folder)/"A1_seed1/model_500.pt"
            checkpoint(previous, 500, 1, "A1", 4)
            process = FakeProcesses()
            with redirect_stdout(io.StringIO()):
                BATCH.run_batch(args, read_checkpoint, process)
            self.assertEqual(len(process.commands), 1)
            command = process.commands[0]
            self.assertEqual(command[command.index("--max_iterations")+1], "2500")
            self.assertEqual(Path(command[command.index("--resume_checkpoint")+1]), previous)
            self.assertTrue(previous.is_file())
            model, iteration = BATCH.latest_checkpoint(previous.parent, args, "A1", 1, read_checkpoint)
            self.assertEqual(iteration, 3000)
            self.assertEqual(model.parent.name, "resume_500_to_3000")

    def test_failure_stops_queue_and_records_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1"])
            process = FakeProcesses(failure=17)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                BATCH.run_batch(args, read_checkpoint, process)
            self.assertEqual(len(process.commands), 1)
            state = BATCH.read_json(Path(folder)/"ablation_batch.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["jobs"]["A0_seed1"]["status"], "train_failed")

    def test_incompatible_existing_run_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1", "--ablations", "A1"])
            checkpoint(Path(folder)/"A1_seed1/model_3000.pt", 3000, 1, "A1", 64)
            process = FakeProcesses()
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                BATCH.run_batch(args, read_checkpoint, process)
            self.assertEqual(process.commands, [])

    def test_over_budget_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1", "--ablations", "A1"])
            checkpoint(Path(folder)/"A1_seed1/model_4000.pt", 4000, 1, "A1", 1024)
            with self.assertRaises(ValueError):
                BATCH.latest_checkpoint(Path(folder)/"A1_seed1", args, "A1", 1, read_checkpoint)

    def test_truncated_checkpoint_falls_back_to_previous(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            args = BATCH.parse_args(["--output_root", folder, "--seeds", "1", "--ablations", "A1"])
            run = Path(folder)/"A1_seed1"
            checkpoint(run/"model_100.pt", 100, 1, "A1", 1024)
            (run/"model_200.pt").write_bytes(b"interrupted write")
            def reader(path):
                if path.name == "model_200.pt":
                    raise EOFError("truncated checkpoint")
                return read_checkpoint(path)
            with redirect_stdout(io.StringIO()):
                path, iteration = BATCH.latest_checkpoint(run, args, "A1", 1, reader)
            self.assertEqual(iteration, 100)
            self.assertEqual(path.name, "model_100.pt")


if __name__ == "__main__":
    unittest.main()
