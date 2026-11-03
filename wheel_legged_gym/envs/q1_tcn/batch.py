"""Run A0--A4 sequentially, resume unfinished training, then evaluate and compare.

Run by file path so planning/help do not import Isaac Gym or PyTorch.
"""

import argparse
from datetime import datetime
import inspect
import json
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
MODULE = "wheel_legged_gym.envs.q1_tcn."
ABLATIONS = ("A0", "A1", "A2", "A3", "A4")
SCENARIOS = ("stand", "tracking", "push_front", "push_back", "push_side", "low_friction", "payload")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=3000, help="Total target iterations per experiment")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--output_root", type=Path, default=Path("logs/q1_tcn"))
    parser.add_argument("--eval_num_envs", type=int, default=64)
    parser.add_argument("--eval_seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--eval_duration", type=float, default=12.0)
    parser.add_argument("--train_only", action="store_true", help="Skip evaluation and comparison")
    parser.add_argument("--dry_run", action="store_true", help="Print the fresh-run plan without running or writing files")
    args = parser.parse_args(argv)
    if min(args.num_envs, args.max_iterations, args.eval_num_envs) < 1:
        parser.error("Environment counts and max_iterations must be positive")
    if not 7 <= args.eval_duration < float("inf"):
        parser.error("eval_duration must be finite and >= 7 seconds")
    for name in ("ablations", "seeds", "eval_seeds"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error("{} must not contain duplicates".format(name))
    if not args.output_root.is_absolute():
        args.output_root = ROOT / args.output_root
    args.output_root = args.output_root.resolve()
    return args


def device_args(args):
    return ["--headless", "--sim_device", args.sim_device, "--rl_device", args.rl_device]


def train_command(args, ablation, seed, output, remaining=None, checkpoint=None):
    command = [sys.executable, "-u", "-m", MODULE+"train", "--ablation", ablation,
        "--seed", str(seed), "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.max_iterations if remaining is None else remaining),
        "--output_dir", str(output)] + device_args(args)
    if checkpoint is not None:
        command += ["--resume_checkpoint", str(checkpoint)]
    return command


def eval_command(args, checkpoint, output):
    return [sys.executable, "-u", "-m", MODULE+"evaluate", "--checkpoint_path", str(checkpoint),
        "--num_envs", str(args.eval_num_envs), "--eval_seeds"] + [str(seed) for seed in args.eval_seeds] + [
        "--duration", str(args.eval_duration), "--output_dir", str(output)] + device_args(args)


def compare_command(args, summaries):
    return [sys.executable, "-u", str(Path(__file__).with_name("compare.py")),
            "--inputs"] + [str(path) for path in summaries] + [
            "--output_dir", str(args.output_root / "comparison")]


def print_command(command):
    # Display only; subprocess always receives an argument list with shell=False.
    print(subprocess.list2cmdline(command), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_info(path):
    import torch
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    saved = torch.load(path, **options)
    return {name: saved[name] for name in ("iteration", "total_steps", "config")}


def validate_saved_run(saved, args, ablation, seed):
    if saved.get("task") != "q1_tcn" or saved.get("schema_version") != 1:
        raise ValueError("Expected a q1_tcn schema_version=1 snapshot")
    if (saved["train"]["ablation"] != ablation or saved["train"]["seed"] != seed
            or saved["env"]["seed"] != seed or saved["env"]["env"]["num_envs"] != args.num_envs):
        raise ValueError("Existing run has a different ablation, seed or num_envs; use a different output_root")


def latest_checkpoint(run_dir, args, ablation, seed, read_checkpoint=checkpoint_info):
    candidates = []
    for path in run_dir.rglob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None, 0
    for iteration, path in sorted(candidates, key=lambda pair: (pair[0], str(pair[1])), reverse=True):
        try:
            info = read_checkpoint(path)
            break
        except (EOFError, RuntimeError, pickle.UnpicklingError) as error:
            print("Cannot read checkpoint {}; trying an earlier checkpoint: {}".format(path, error), flush=True)
    else:
        raise ValueError("No readable checkpoint in {}; preserve the files and inspect the training output".format(run_dir))
    validate_saved_run(info["config"], args, ablation, seed)
    expected_steps = iteration*args.num_envs*info["config"]["train"]["runner"]["num_steps_per_env"]
    if info["iteration"] != iteration or info["total_steps"] != expected_steps:
        raise ValueError("Checkpoint counters disagree with its filename/budget: {}".format(path))
    if iteration > args.max_iterations:
        raise ValueError("Existing checkpoint exceeds the requested training budget: {}".format(path))
    return path, iteration


def unused_directory(base):
    if not base.exists() or not any(base.iterdir()):
        return base
    index = 1
    while True:
        path = base.with_name(base.name+"_attempt{:03d}".format(index))
        if not path.exists():
            return path
        index += 1


def matching_evaluation(run_dir, checkpoint, args, seed, ablation):
    for summary in sorted(run_dir.rglob("summary.json")):
        try:
            saved = read_json(summary)
        except (ValueError, OSError):
            continue
        protocol = saved.get("evaluation", {})
        if (Path(saved.get("checkpoint", "")).resolve() != checkpoint.resolve()
                or saved.get("training_seed") != seed or saved.get("ablation") != ablation
                or protocol.get("seeds") != args.eval_seeds
                or protocol.get("num_envs") != args.eval_num_envs
                or protocol.get("duration_s") != args.eval_duration
                or protocol.get("pulse_force_n") != 8.0
                or set(saved.get("scenarios", {})) != set(SCENARIOS)):
            continue
        files = ["summary.csv", "trials.csv"] + [scenario+".png" for scenario in SCENARIOS]
        if all((summary.parent / name).is_file() for name in files):
            return summary
    return None


def save_state(path, state):
    state["updated_at"] = datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute(command, job, stage, state, state_path, run_process=subprocess.run):
    state["status"] = "running"
    job["status"] = stage+"_running"
    job["command"] = command
    save_state(state_path, state)
    print_command(command)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        result = run_process(command, cwd=str(ROOT), env=environment, check=False)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        job["status"] = stage+"_interrupted"
        save_state(state_path, state)
        raise
    except OSError as error:
        state["status"] = "failed"
        job["status"], job["error"] = stage+"_failed", str(error)
        save_state(state_path, state)
        raise
    if result.returncode:
        state["status"] = "failed"
        job["status"], job["returncode"] = stage+"_failed", result.returncode
        save_state(state_path, state)
        raise RuntimeError("{} exited with code {}; batch stopped".format(stage, result.returncode))
    job["status"] = stage+"_finished"
    save_state(state_path, state)


def run_batch(args, read_checkpoint=checkpoint_info, run_process=subprocess.run):
    total = len(args.seeds)*len(args.ablations)
    print("{} experiments; sequential execution; target {} iterations each".format(total, args.max_iterations), flush=True)
    if args.dry_run:
        summaries = []
        for seed in args.seeds:
            for ablation in args.ablations:
                run_dir = args.output_root / "{}_seed{}".format(ablation, seed)
                print_command(train_command(args, ablation, seed, run_dir))
                if not args.train_only:
                    checkpoint = run_dir / "model_{}.pt".format(args.max_iterations)
                    print_command(eval_command(args, checkpoint, run_dir / "evaluation"))
                    summaries.append(run_dir / "evaluation" / "summary.json")
        if summaries:
            print_command(compare_command(args, summaries))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "ablation_batch.json"
    state = {"jobs": {}, "status": "running"}
    if state_path.exists():
        state["jobs"] = read_json(state_path).get("jobs", {})
    summaries = []
    for seed in args.seeds:
        for ablation in args.ablations:
            key = "{}_seed{}".format(ablation, seed)
            print("\n=== {} ===".format(key), flush=True)
            run_dir = args.output_root / key
            job = state["jobs"].setdefault(key, {})
            snapshot = run_dir / "resolved_config.json"
            if snapshot.exists():
                validate_saved_run(read_json(snapshot), args, ablation, seed)
            checkpoint, iteration = latest_checkpoint(run_dir, args, ablation, seed, read_checkpoint)
            if iteration < args.max_iterations:
                remaining = args.max_iterations-iteration
                output = run_dir if checkpoint is None and not run_dir.exists() else unused_directory(
                    run_dir / ("resume_{}_to_{}".format(iteration, args.max_iterations) if checkpoint else "restart"))
                execute(train_command(args, ablation, seed, output, remaining, checkpoint),
                        job, "train", state, state_path, run_process)
                checkpoint, iteration = latest_checkpoint(run_dir, args, ablation, seed, read_checkpoint)
                if iteration != args.max_iterations:
                    raise RuntimeError("Training exited without the expected final checkpoint for {}".format(key))
            else:
                print("Training complete; reuse {}".format(checkpoint), flush=True)
            job["checkpoint"] = str(checkpoint)
            if not args.train_only:
                summary = matching_evaluation(run_dir, checkpoint, args, seed, ablation)
                if summary is None:
                    output = unused_directory(checkpoint.parent / "evaluation")
                    execute(eval_command(args, checkpoint, output), job, "evaluate", state, state_path, run_process)
                    summary = matching_evaluation(run_dir, checkpoint, args, seed, ablation)
                    if summary is None:
                        raise RuntimeError("Evaluation exited without complete matching results for {}".format(key))
                else:
                    print("Evaluation complete; reuse {}".format(summary), flush=True)
                summaries.append(summary)
                job["summary"] = str(summary)
            job["status"] = "complete"
            save_state(state_path, state)
    if summaries:
        execute(compare_command(args, summaries), state.setdefault("comparison", {}),
                "compare", state, state_path, run_process)
    state["status"] = "complete"
    save_state(state_path, state)
    print("Batch complete: {}".format(args.output_root), flush=True)


def main():
    args = parse_args()
    try:
        run_batch(args)
    except KeyboardInterrupt:
        print("Batch interrupted. Run the same command to continue from saved checkpoints.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as error:
        state_path = args.output_root / "ablation_batch.json"
        if not args.dry_run and state_path.exists():
            state = read_json(state_path)
            state["status"], state["error"] = "failed", str(error)
            save_state(state_path, state)
        print("Batch stopped: {}".format(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
