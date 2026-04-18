#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from peorl.linear import make_linear_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["linear_bandit", "linear_branching", "linear_intrinsic", "linear_near_intrinsic"],
        default="linear_bandit",
    )
    parser.add_argument("--feature-dim", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = make_linear_task(args.task, feature_dim=args.feature_dim)

    print(f"Task: {task.mdp.name}")
    print(task.description)
    print(f"Horizon: {task.mdp.horizon}")
    print(f"States: {task.mdp.num_states}")
    print(f"Actions: {task.mdp.num_actions}")
    print(f"Feature dim: {task.mdp.feature_dim}")
    print("Sample feature vectors:")
    for state in range(min(task.mdp.num_states, 3)):
        for action in range(min(task.mdp.num_actions, 2)):
            print(f"  state={state} action={action} feature={task.mdp.features[state, action].tolist()}")


if __name__ == "__main__":
    main()
