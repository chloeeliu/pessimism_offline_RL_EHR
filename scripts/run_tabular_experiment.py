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

from peorl.envs import make_task
from peorl.experiments import RunMetrics, run_single_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["bandit", "branching"], default="branching")
    parser.add_argument("--dataset-sizes", default="20,50,100,200,500")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds.")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--support-mask-threshold", type=int, default=None)
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metrics, summary JSON, and plots. Defaults to results/raw/<task>.",
    )
    return parser.parse_args()


DEFAULTS = {
    "task": "branching",
    "dataset_sizes": "20,50,100,200,500",
    "seeds": 50,
    "beta": 1.0,
    "support_mask_threshold": None,
    "config": None,
    "output_dir": None,
}


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        args.task_params = {}
        return args

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for key, value in config.items():
        if key == "task_params":
            continue
        normalized_key = key.replace("-", "_")
        if getattr(args, normalized_key, None) == DEFAULTS.get(normalized_key):
            setattr(args, normalized_key, value)
    args.task_params = config.get("task_params", {})
    return args


def summarize(metrics: list[RunMetrics]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, list[RunMetrics]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        grouped[row.method][row.num_episodes].append(row)

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for method, by_size in grouped.items():
        summary[method] = {}
        for num_episodes, rows in sorted(by_size.items()):
            values = [row.policy_value for row in rows]
            low_support = [row.low_support_mass for row in rows]
            root_counts = [row.root_action_count for row in rows]
            q_errors = [row.chosen_action_q_error for row in rows]
            agreements = [row.action_agreement_mass for row in rows]
            bonuses = [row.chosen_action_bonus for row in rows]
            summary[method][num_episodes] = {
                "mean_value": mean(values),
                "std_value": pstdev(values) if len(values) > 1 else 0.0,
                "mean_low_support_mass": mean(low_support),
                "mean_root_action_count": mean(root_counts),
                "mean_chosen_action_q_error": mean(q_errors),
                "mean_action_agreement_mass": mean(agreements),
                "mean_chosen_action_bonus": mean(bonuses),
            }
    return summary


def save_metrics(output_dir: Path, metrics: list[RunMetrics], summary: dict, resolved_config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = list(metrics[0].to_dict().keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.to_dict())

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    config_path = output_dir / "resolved_config.json"
    with config_path.open("w") as handle:
        json.dump(resolved_config, handle, indent=2)


def make_plots(output_dir: Path, metrics: list[RunMetrics]) -> None:
    grouped: dict[str, dict[int, list[RunMetrics]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        grouped[row.method][row.num_episodes].append(row)

    sizes = sorted({row.num_episodes for row in metrics})
    methods = sorted({row.method for row in metrics})
    colors = {"greedy": "#c03d3d", "pessimistic": "#1f6f8b", "support_masked": "#2f9e44"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for method in methods:
        value_means = [mean([row.policy_value for row in grouped[method][size]]) for size in sizes]
        value_stds = [pstdev([row.policy_value for row in grouped[method][size]]) if len(grouped[method][size]) > 1 else 0.0 for size in sizes]
        axes[0, 0].plot(sizes, value_means, marker="o", label=method, color=colors[method])
        axes[0, 0].fill_between(
            sizes,
            [m - s for m, s in zip(value_means, value_stds)],
            [m + s for m, s in zip(value_means, value_stds)],
            alpha=0.15,
            color=colors[method],
        )

        support_means = [mean([row.low_support_mass for row in grouped[method][size]]) for size in sizes]
        q_error_means = [mean([row.chosen_action_q_error for row in grouped[method][size]]) for size in sizes]
        agreement_means = [mean([row.action_agreement_mass for row in grouped[method][size]]) for size in sizes]
        axes[0, 1].plot(sizes, support_means, marker="o", label=method, color=colors[method])
        axes[1, 0].plot(sizes, q_error_means, marker="o", label=method, color=colors[method])
        axes[1, 1].plot(sizes, agreement_means, marker="o", label=method, color=colors[method])

    axes[0, 0].set_title("True Policy Value")
    axes[0, 0].set_xlabel("Offline episodes")
    axes[0, 0].set_ylabel("Exact return")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("Mass on Low-Support Actions")
    axes[0, 1].set_xlabel("Offline episodes")
    axes[0, 1].set_ylabel("Occupancy mass")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title("Chosen-Action Q Overestimation")
    axes[1, 0].set_xlabel("Offline episodes")
    axes[1, 0].set_ylabel("Mean q_hat - q_true")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title("Agreement with Optimal Policy")
    axes[1, 1].set_xlabel("Offline episodes")
    axes[1, 1].set_ylabel("Occupancy-weighted agreement")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = apply_config(parse_args())
    dataset_sizes = [int(token) for token in args.dataset_sizes.split(",") if token]
    task_params = getattr(args, "task_params", {})
    task = make_task(args.task, **task_params)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "raw" / args.task

    metrics: list[RunMetrics] = []
    for num_episodes in dataset_sizes:
        for seed in range(args.seeds):
            metrics.extend(
                run_single_seed(
                    task=task,
                    num_episodes=num_episodes,
                    seed=seed,
                    beta=args.beta,
                    support_mask_threshold=args.support_mask_threshold,
                )
            )

    summary = summarize(metrics)
    resolved_config = {
        "task": args.task,
        "dataset_sizes": dataset_sizes,
        "seeds": args.seeds,
        "beta": args.beta,
        "support_mask_threshold": args.support_mask_threshold,
        "output_dir": str(output_dir),
        "task_params": task_params,
    }
    save_metrics(output_dir, metrics, summary, resolved_config)
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
