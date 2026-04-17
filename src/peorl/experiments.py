from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from peorl.algorithms import fit_greedy_plugin, fit_pessimistic_vi, fit_support_masked_vi
from peorl.datasets import build_empirical_model, collect_dataset
from peorl.envs.tabular import OfflineTask
from peorl.evaluation import (
    compute_action_agreement_mass,
    compute_policy_q_values,
    compute_state_occupancy,
    compute_support_stats,
    evaluate_policy,
    solve_optimal_policy,
)


@dataclass(frozen=True)
class RunMetrics:
    task_name: str
    method: str
    seed: int
    num_episodes: int
    beta: float
    policy_value: float
    behavior_value: float
    optimal_value: float
    suboptimality_gap: float
    root_action: int
    root_action_count: float
    expected_chosen_count: float
    low_support_mass: float
    action_agreement_mass: float
    chosen_action_q_error: float
    chosen_action_bonus: float
    root_true_q: float
    root_q_hat: float
    root_q_value: float
    root_bonus: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def run_single_seed(
    task: OfflineTask,
    num_episodes: int,
    seed: int,
    beta: float,
    support_mask_threshold: int | None = None,
) -> list[RunMetrics]:
    dataset = collect_dataset(
        mdp=task.mdp,
        behavior_policy=task.behavior_policy,
        num_episodes=num_episodes,
        seed=seed,
    )
    model = build_empirical_model(task.mdp, dataset)

    optimal_policy, _ = solve_optimal_policy(task.mdp)
    optimal_value = evaluate_policy(task.mdp, optimal_policy)
    behavior_value = evaluate_policy(task.mdp, task.behavior_policy)

    runs = [
        ("greedy", fit_greedy_plugin(task.mdp, model)),
        ("pessimistic", fit_pessimistic_vi(task.mdp, model, beta=beta)),
    ]
    if support_mask_threshold is not None:
        runs.append(("support_masked", fit_support_masked_vi(task.mdp, model, min_count=support_mask_threshold)))

    metrics: list[RunMetrics] = []
    for method_name, planner_result in runs:
        policy_value = evaluate_policy(task.mdp, planner_result.policy)
        true_q = compute_policy_q_values(task.mdp, planner_result.policy)
        occupancy = compute_state_occupancy(task.mdp, planner_result.policy)
        greedy_actions = np.argmax(planner_result.policy, axis=2)
        chosen_q_error = 0.0
        chosen_bonus = 0.0
        for step in range(task.mdp.horizon):
            chosen_hat = planner_result.q_hat[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_true = true_q[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_bonus_step = planner_result.bonuses[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_q_error += float(np.dot(occupancy[step], chosen_hat - chosen_true))
            chosen_bonus += float(np.dot(occupancy[step], chosen_bonus_step))

        support = compute_support_stats(
            mdp=task.mdp,
            policy=planner_result.policy,
            counts=model.counts,
            support_threshold=task.support_threshold,
        )
        root_state = int(np.argmax(task.mdp.initial_state_dist))
        root_action = support.root_action
        metrics.append(
            RunMetrics(
                task_name=task.mdp.name,
                method=method_name,
                seed=seed,
                num_episodes=num_episodes,
                beta=beta,
                policy_value=policy_value,
                behavior_value=behavior_value,
                optimal_value=optimal_value,
                suboptimality_gap=optimal_value - policy_value,
                root_action=support.root_action,
                root_action_count=support.root_action_count,
                expected_chosen_count=support.expected_chosen_count,
                low_support_mass=support.low_support_mass,
                action_agreement_mass=compute_action_agreement_mass(task.mdp, planner_result.policy, optimal_policy),
                chosen_action_q_error=chosen_q_error / float(task.mdp.horizon),
                chosen_action_bonus=chosen_bonus / float(task.mdp.horizon),
                root_true_q=float(true_q[0, root_state, root_action]),
                root_q_hat=float(planner_result.q_hat[0, root_state, root_action]),
                root_q_value=float(planner_result.q_values[0, root_state, root_action]),
                root_bonus=float(planner_result.bonuses[0, root_state, root_action]),
            )
        )

    return metrics
