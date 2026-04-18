#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

from peorl.experiments import RunMetrics, run_single_seed
from peorl.envs import make_task


DEFAULTS = {
    "task": "branching",
    "dataset_sizes": "20,50,100,200,500",
    "seeds": 50,
    "support_mask_threshold": None,
    "betas": "0.0,0.2,0.4,0.8,1.2,1.6,2.0",
    "config": None,
    "output_dir": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["bandit", "branching", "intrinsic", "near_intrinsic"], default="branching")
    parser.add_argument("--dataset-sizes", default="20,50,100,200,500")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--support-mask-threshold", type=int, default=None)
    parser.add_argument("--betas", default="0.0,0.2,0.4,0.8,1.2,1.6,2.0")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


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


def summarize_rows(rows: list[RunMetrics]) -> dict[str, float]:
    values = [row.policy_value for row in rows]
    low_support = [row.low_support_mass for row in rows]
    q_errors = [row.chosen_action_q_error for row in rows]
    agreements = [row.action_agreement_mass for row in rows]
    bonuses = [row.chosen_action_bonus for row in rows]
    return {
        "mean_value": mean(values),
        "std_value": pstdev(values) if len(values) > 1 else 0.0,
        "mean_low_support_mass": mean(low_support),
        "mean_chosen_action_q_error": mean(q_errors),
        "mean_action_agreement_mass": mean(agreements),
        "mean_chosen_action_bonus": mean(bonuses),
    }


def summarize(
    baseline_rows: list[RunMetrics],
    pessimistic_rows: list[RunMetrics],
    betas: list[float],
    dataset_sizes: list[int],
) -> dict[str, object]:
    baselines: dict[str, dict[str, dict[str, float]]] = {}
    for method in sorted({row.method for row in baseline_rows}):
        baselines[method] = {}
        for size in dataset_sizes:
            rows = [row for row in baseline_rows if row.method == method and row.num_episodes == size]
            baselines[method][str(size)] = summarize_rows(rows)

    pessimistic: dict[str, dict[str, dict[str, float]]] = {}
    for beta in betas:
        beta_key = f"{beta:.3f}"
        pessimistic[beta_key] = {}
        for size in dataset_sizes:
            rows = [
                row
                for row in pessimistic_rows
                if row.method == "pessimistic" and row.num_episodes == size and abs(row.beta - beta) < 1e-12
            ]
            pessimistic[beta_key][str(size)] = summarize_rows(rows)

    best_beta_by_size: dict[str, dict[str, float]] = {}
    for size in dataset_sizes:
        best_beta = None
        best_stats = None
        for beta in betas:
            stats = pessimistic[f"{beta:.3f}"][str(size)]
            if best_stats is None or stats["mean_value"] > best_stats["mean_value"]:
                best_beta = beta
                best_stats = stats
        greedy_stats = baselines["greedy"][str(size)]
        best_beta_by_size[str(size)] = {
            "best_beta": float(best_beta),
            "best_value": float(best_stats["mean_value"]),
            "best_improvement_over_greedy": float(best_stats["mean_value"] - greedy_stats["mean_value"]),
            "best_low_support_mass": float(best_stats["mean_low_support_mass"]),
            "best_q_error": float(best_stats["mean_chosen_action_q_error"]),
        }

    return {
        "baselines": baselines,
        "pessimistic": pessimistic,
        "best_beta_by_size": best_beta_by_size,
    }


def save_outputs(
    output_dir: Path,
    baseline_rows: list[RunMetrics],
    pessimistic_rows: list[RunMetrics],
    summary: dict[str, object],
    resolved_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = list((baseline_rows + pessimistic_rows)[0].to_dict().keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in baseline_rows + pessimistic_rows:
            writer.writerow(row.to_dict())

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "resolved_config.json").open("w") as handle:
        json.dump(resolved_config, handle, indent=2)


def _subplot_grid(n_panels: int) -> tuple[int, int]:
    cols = 3 if n_panels > 3 else n_panels
    rows = math.ceil(n_panels / cols)
    return rows, cols


def make_value_panels(
    output_dir: Path,
    summary: dict[str, object],
    dataset_sizes: list[int],
    betas: list[float],
) -> None:
    baselines = summary["baselines"]
    pessimistic = summary["pessimistic"]
    rows, cols = _subplot_grid(len(dataset_sizes))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for index, size in enumerate(dataset_sizes):
        ax = axes[index // cols][index % cols]
        beta_keys = [f"{beta:.3f}" for beta in betas]
        means = [pessimistic[beta_key][str(size)]["mean_value"] for beta_key in beta_keys]
        stds = [pessimistic[beta_key][str(size)]["std_value"] for beta_key in beta_keys]

        ax.plot(betas, means, marker="o", color="#1f6f8b", label="pessimistic")
        ax.fill_between(
            betas,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color="#1f6f8b",
            alpha=0.18,
        )
        ax.axhline(
            baselines["greedy"][str(size)]["mean_value"],
            color="#c03d3d",
            linestyle="--",
            label="greedy",
        )
        if "support_masked" in baselines:
            ax.axhline(
                baselines["support_masked"][str(size)]["mean_value"],
                color="#2f9e44",
                linestyle=":",
                label="support_masked",
            )
        ax.set_title(f"Episodes = {size}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Mean value")
        ax.grid(alpha=0.3)
        if index == 0:
            ax.legend()

    for index in range(len(dataset_sizes), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "beta_sweep_values.png", dpi=180)
    plt.close(fig)


def make_diagnostic_panels(
    output_dir: Path,
    summary: dict[str, object],
    dataset_sizes: list[int],
    betas: list[float],
) -> None:
    pessimistic = summary["pessimistic"]
    beta_keys = [f"{beta:.3f}" for beta in betas]
    colors = ["#0b7285", "#1971c2", "#5f3dc4", "#a61e4d", "#e67700", "#2b8a3e"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    diagnostics = [
        ("mean_low_support_mass", "Low-Support Mass", "Occupancy mass"),
        ("mean_chosen_action_q_error", "Chosen-Action Q Error", "Mean q_hat - q_true"),
        ("mean_action_agreement_mass", "Optimal Policy Agreement", "Agreement"),
    ]

    for size, color in zip(dataset_sizes, colors):
        for ax, (metric_key, title, ylabel) in zip(axes, diagnostics):
            values = [pessimistic[beta_key][str(size)][metric_key] for beta_key in beta_keys]
            ax.plot(betas, values, marker="o", label=f"{size} episodes", color=color)
            ax.set_title(title)
            ax.set_xlabel("beta")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    axes[2].set_ylim(0.0, 1.05)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "beta_sweep_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = apply_config(parse_args())
    dataset_sizes = [int(token) for token in args.dataset_sizes.split(",") if token]
    betas = [float(token) for token in args.betas.split(",") if token]
    task_params = getattr(args, "task_params", {})
    task = make_task(args.task, **task_params)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "raw" / f"{args.task}_beta_sweep"

    baseline_rows: list[RunMetrics] = []
    pessimistic_rows: list[RunMetrics] = []
    baseline_emitted = False

    for beta in betas:
        for num_episodes in dataset_sizes:
            for seed in range(args.seeds):
                run_rows = run_single_seed(
                    task=task,
                    num_episodes=num_episodes,
                    seed=seed,
                    beta=beta,
                    support_mask_threshold=args.support_mask_threshold,
                )
                pessimistic_rows.extend([row for row in run_rows if row.method == "pessimistic"])
                if not baseline_emitted:
                    baseline_rows.extend([row for row in run_rows if row.method != "pessimistic"])
        baseline_emitted = True

    summary = summarize(
        baseline_rows=baseline_rows,
        pessimistic_rows=pessimistic_rows,
        betas=betas,
        dataset_sizes=dataset_sizes,
    )
    resolved_config = {
        "task": args.task,
        "dataset_sizes": dataset_sizes,
        "seeds": args.seeds,
        "support_mask_threshold": args.support_mask_threshold,
        "betas": betas,
        "output_dir": str(output_dir),
        "task_params": task_params,
    }
    save_outputs(output_dir, baseline_rows, pessimistic_rows, summary, resolved_config)
    make_value_panels(output_dir, summary, dataset_sizes, betas)
    make_diagnostic_panels(output_dir, summary, dataset_sizes, betas)

    print(f"Task: {task.mdp.name}")
    print(f"Wrote metrics to {output_dir / 'metrics.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote plots to {output_dir / 'beta_sweep_values.png'} and {output_dir / 'beta_sweep_diagnostics.png'}")
    print("\nBest beta by dataset size:")
    for size in dataset_sizes:
        stats = summary["best_beta_by_size"][str(size)]
        print(
            f"  episodes={size:>4}  best_beta={stats['best_beta']:.3f}  "
            f"value={stats['best_value']:.3f}  improvement_over_greedy={stats['best_improvement_over_greedy']:.3f}"
        )


if __name__ == "__main__":
    main()
