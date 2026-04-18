#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peorl.linear import make_linear_task
from peorl.linear.experiments import LinearRunMetrics, run_single_linear_seed


DEFAULTS = {
    "task": "linear_branching",
    "dataset_sizes": "20,50,100,200,500",
    "seeds": 50,
    "ridge": 1.0,
    "beta": 0.8,
    "support_mask_threshold": None,
    "config": None,
    "output_dir": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["linear_bandit", "linear_branching", "linear_intrinsic", "linear_near_intrinsic"],
        default="linear_branching",
    )
    parser.add_argument("--dataset-sizes", default="20,50,100,200,500")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--support-mask-threshold", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        args.task_params = {}
        return args

    config = json.loads(Path(args.config).read_text())
    for key, value in config.items():
        if key == "task_params":
            continue
        normalized_key = key.replace("-", "_")
        if getattr(args, normalized_key, None) == DEFAULTS.get(normalized_key):
            setattr(args, normalized_key, value)
    args.task_params = config.get("task_params", {})
    return args


def summarize(metrics: list[LinearRunMetrics]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, list[LinearRunMetrics]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        grouped[row.method][row.num_episodes].append(row)

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for method, by_size in grouped.items():
        summary[method] = {}
        for num_episodes, rows in sorted(by_size.items()):
            summary[method][num_episodes] = {
                "mean_value": mean([row.policy_value for row in rows]),
                "std_value": pstdev([row.policy_value for row in rows]) if len(rows) > 1 else 0.0,
                "mean_low_support_mass": mean([row.low_support_mass for row in rows]),
                "mean_chosen_action_q_error": mean([row.chosen_action_q_error for row in rows]),
                "mean_action_agreement_mass": mean([row.action_agreement_mass for row in rows]),
                "mean_chosen_action_bonus": mean([row.chosen_action_bonus for row in rows]),
                "mean_chosen_feature_novelty": mean([row.chosen_feature_novelty for row in rows]),
            }
    return summary


def save_outputs(
    output_dir: Path,
    metrics: list[LinearRunMetrics],
    summary: dict[str, object],
    resolved_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = list(metrics[0].to_dict().keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.to_dict())

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "resolved_config.json").open("w") as handle:
        json.dump(resolved_config, handle, indent=2)


def make_plots(output_dir: Path, metrics: list[LinearRunMetrics]) -> None:
    grouped: dict[str, dict[int, list[LinearRunMetrics]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        grouped[row.method][row.num_episodes].append(row)

    sizes = sorted({row.num_episodes for row in metrics})
    methods = sorted({row.method for row in metrics})
    colors = {"greedy": "#c03d3d", "pessimistic": "#1f6f8b", "support_masked": "#2f9e44"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    metric_specs = [
        ("policy_value", "True Policy Value", "Exact return"),
        ("low_support_mass", "Mass on Low-Support Actions", "Occupancy mass"),
        ("chosen_action_q_error", "Chosen-Action Q Overestimation", "Mean q_hat - q_true"),
        ("action_agreement_mass", "Agreement with Optimal Policy", "Agreement"),
        ("chosen_action_bonus", "Chosen-Action Bonus", "Mean bonus"),
        ("chosen_feature_novelty", "Chosen Feature Novelty", "Mean novelty proxy"),
    ]

    for method in methods:
        rows_by_size = grouped[method]
        for ax, (field, title, ylabel) in zip(axes.flat, metric_specs):
            means = [mean([getattr(row, field) for row in rows_by_size[size]]) for size in sizes]
            stds = [pstdev([getattr(row, field) for row in rows_by_size[size]]) if len(rows_by_size[size]) > 1 else 0.0 for size in sizes]
            ax.plot(sizes, means, marker="o", label=method, color=colors[method])
            ax.fill_between(
                sizes,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15,
                color=colors[method],
            )
            ax.set_title(title)
            ax.set_xlabel("Offline episodes")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    axes[1, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = apply_config(parse_args())
    dataset_sizes = [int(token) for token in args.dataset_sizes.split(",") if token]
    task_params = getattr(args, "task_params", {})
    task = make_linear_task(args.task, **task_params)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "raw" / args.task

    metrics: list[LinearRunMetrics] = []
    for num_episodes in dataset_sizes:
        for seed in range(args.seeds):
            metrics.extend(
                run_single_linear_seed(
                    task=task,
                    num_episodes=num_episodes,
                    seed=seed,
                    ridge=args.ridge,
                    beta=args.beta,
                    support_mask_threshold=args.support_mask_threshold,
                )
            )

    summary = summarize(metrics)
    resolved_config = {
        "task": args.task,
        "dataset_sizes": dataset_sizes,
        "seeds": args.seeds,
        "ridge": args.ridge,
        "beta": args.beta,
        "support_mask_threshold": args.support_mask_threshold,
        "output_dir": str(output_dir),
        "task_params": task_params,
    }
    save_outputs(output_dir, metrics, summary, resolved_config)
    make_plots(output_dir, metrics)

    print(f"Task: {task.mdp.name}")
    print(task.description)
    print(f"Wrote metrics to {output_dir / 'metrics.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote plot to {output_dir / 'comparison.png'}")
    for method, by_size in summary.items():
        print(f"\n{method}:")
        for size, stats in by_size.items():
            print(
                f"  episodes={size:>4}  value={stats['mean_value']:.3f} +/- {stats['std_value']:.3f}  "
                f"low_support_mass={stats['mean_low_support_mass']:.3f}  "
                f"q_error={stats['mean_chosen_action_q_error']:.3f}  "
                f"agreement={stats['mean_action_agreement_mass']:.3f}"
            )


if __name__ == "__main__":
    main()
