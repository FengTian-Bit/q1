"""Compare evaluation summaries across independent training seeds; no Gym needed."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np


def comparison_rows(reports):
    if not reports:
        raise ValueError("At least one summary.json is required")
    reference = reports[0]
    protocol = copy.deepcopy(reference["training_protocol"])
    protocol["env"].pop("seed", None)
    seen = set()
    for report in reports:
        other = copy.deepcopy(report["training_protocol"])
        other["env"].pop("seed", None)
        if (report["evaluation"] != reference["evaluation"] or other != protocol
                or report["training_steps"] != reference["training_steps"]
                or set(report["scenarios"]) != set(reference["scenarios"])):
            raise ValueError("Comparison requires matching training budgets, conditions and evaluation protocols")
        key = (report["ablation"], report["training_seed"])
        if key in seen:
            raise ValueError("Duplicate training seed for {}: {}".format(*key))
        seen.add(key)
    rows = []
    for ablation in sorted({report["ablation"] for report in reports}):
        group = [report for report in reports if report["ablation"] == ablation]
        for scenario in sorted(reference["scenarios"]):
            names = reference["scenarios"][scenario]["metrics"]
            for name in names:
                values = [report["scenarios"][scenario]["metrics"][name]["mean"] for report in group]
                values = np.array([value for value in values if value is not None], dtype=float)
                rows.append(dict(ablation=ablation, scenario=scenario, metric=name,
                    training_seeds=len(group), valid_training_seeds=len(values),
                    mean=float(values.mean()) if len(values) else None,
                    std_across_training_seeds=float(values.std(ddof=1)) if len(values) > 1 else None))
    return rows


def plot_comparison(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = (("failed", "failure fraction"), ("speed_rmse_m_s", "vx RMSE [m/s]"),
               ("pitch_rms_rad", "pitch RMS [rad]"), ("contact_slip_mae_m_s", "supported slip [m/s]"),
               ("actuator_saturation_fraction", "actuator saturation fraction"), ("recovered", "recovery fraction"))
    groups = sorted({row["ablation"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    width = 0.8/len(groups)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for index, group in enumerate(groups):
            selected = {row["scenario"]: row for row in rows
                        if row["ablation"] == group and row["metric"] == metric}
            values = [selected.get(name, {}).get("mean") for name in scenarios]
            values = [np.nan if value is None else value for value in values]
            errors = [selected.get(name, {}).get("std_across_training_seeds") or 0 for name in scenarios]
            x = np.arange(len(scenarios)) + (index-(len(groups)-1)/2)*width
            axis.bar(x, values, width, yerr=errors, label=group, capsize=2)
        axis.set_xticks(np.arange(len(scenarios)))
        axis.set_xticklabels(scenarios, rotation=30, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Q1 TCN ablations | error bars: SD across training seeds (when >= 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Evaluation summary.json paths")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    rows = comparison_rows(reports)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    plot_comparison(output / "comparison.png", rows)
    print("Comparison: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
