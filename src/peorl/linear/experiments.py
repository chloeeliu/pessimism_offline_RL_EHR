from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from peorl.linear.algorithms import LinearFitConfig, fit_linear_pevi, fit_linear_support_masked_vi
from peorl.linear.data import collect_linear_dataset, count_state_actions
from peorl.linear.evaluation import (
    compute_linear_action_agreement_mass,
    compute_linear_policy_q_values,
    compute_linear_state_occupancy,
    compute_linear_support_stats,
    evaluate_linear_policy,
    solve_linear_task_optimal_policy,
)
from peorl.linear.types import LinearOfflineTask


@dataclass(frozen=True)
class LinearRunMetrics:
    task_name: str
    method: str
    seed: int
    num_episodes: int
    beta: float
    ridge: float
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
    chosen_feature_novelty: float
    root_true_q: float
    root_q_hat: float
    root_q_value: float
    root_bonus: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def run_single_linear_seed(
    task: LinearOfflineTask,
    num_episodes: int,
    seed: int,
    ridge: float,
    beta: float,
    support_mask_threshold: int | None = None,
) -> list[LinearRunMetrics]:
    dataset = collect_linear_dataset(
        mdp=task.mdp,
        behavior_policy=task.behavior_policy,
        num_episodes=num_episodes,
        seed=seed,
    )
    counts = count_state_actions(task.mdp, dataset)

    optimal_policy, _ = solve_linear_task_optimal_policy(task.mdp)
    optimal_value = evaluate_linear_policy(task.mdp, optimal_policy)
    behavior_value = evaluate_linear_policy(task.mdp, task.behavior_policy)

    runs = [
        ("greedy", fit_linear_pevi(task.mdp, dataset, LinearFitConfig(ridge=ridge, beta=0.0))),
        ("pessimistic", fit_linear_pevi(task.mdp, dataset, LinearFitConfig(ridge=ridge, beta=beta))),
    ]
    if support_mask_threshold is not None:
        runs.append(
            (
                "support_masked",
                fit_linear_support_masked_vi(task.mdp, dataset, ridge=ridge, min_count=support_mask_threshold),
            )
        )

    metrics: list[LinearRunMetrics] = []
    for method_name, planner_result in runs:
        policy_value = evaluate_linear_policy(task.mdp, planner_result.policy)
        true_q = compute_linear_policy_q_values(task.mdp, planner_result.policy)
        occupancy = compute_linear_state_occupancy(task.mdp, planner_result.policy)
        greedy_actions = np.argmax(planner_result.policy, axis=2)
        chosen_q_error = 0.0
        chosen_bonus = 0.0
        for step in range(task.mdp.horizon):
            chosen_hat = planner_result.q_hat[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_true = true_q[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_bonus_step = planner_result.bonuses[step, np.arange(task.mdp.num_states), greedy_actions[step]]
            chosen_q_error += float(np.dot(occupancy[step], chosen_hat - chosen_true))
            chosen_bonus += float(np.dot(occupancy[step], chosen_bonus_step))

        support = compute_linear_support_stats(
            mdp=task.mdp,
            policy=planner_result.policy,
            counts=counts,
            support_threshold=task.support_threshold,
            bonuses=planner_result.bonuses,
        )
        root_state = int(np.argmax(task.mdp.initial_state_dist))
        root_action = support.root_action
        metrics.append(
            LinearRunMetrics(
                task_name=task.mdp.name,
                method=method_name,
                seed=seed,
                num_episodes=num_episodes,
                beta=beta,
                ridge=ridge,
                policy_value=policy_value,
                behavior_value=behavior_value,
                optimal_value=optimal_value,
                suboptimality_gap=optimal_value - policy_value,
                root_action=root_action,
                root_action_count=support.root_action_count,
                expected_chosen_count=support.expected_chosen_count,
                low_support_mass=support.low_support_mass,
                action_agreement_mass=compute_linear_action_agreement_mass(task.mdp, planner_result.policy, optimal_policy),
                chosen_action_q_error=chosen_q_error / float(task.mdp.horizon),
                chosen_action_bonus=chosen_bonus / float(task.mdp.horizon),
                chosen_feature_novelty=support.chosen_feature_novelty,
                root_true_q=float(true_q[0, root_state, root_action]),
                root_q_hat=float(planner_result.q_hat[0, root_state, root_action]),
                root_q_value=float(planner_result.q_values[0, root_state, root_action]),
                root_bonus=float(planner_result.bonuses[0, root_state, root_action]),
            )
        )

    return metrics
